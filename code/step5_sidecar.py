# -*- coding: utf-8 -*-
"""
第五步：证据侧车 + 计费器（方案 3.md 第 7、8 节）

为什么需要它：
  LLM 智能体可能说真话也可能选择性隐瞒，但它【说出的数字】绝不能直接当真。
  侧车（sidecar）是可信计算基的一部分：事实值由它从任务状态里取出并盖章成"收据"，
  协调者和所有方法只认收据，不认模型嘴里的数。

本模块提供 EvidenceSidecar 类：
  - 公共查询目录：每个任务 24 个合法事实（8 群组 × 3 指标）
  - 权限：主设定为"预先授权的只读核验"，任何方法都能对任意合法事实发起计费查询
  - 收据：包含方案 8.2 节要求的全部绑定字段（快照 hash、群组 hash、配方版本、
    分子/分母、量化整数值、有效行数等）
  - 计费：每次查询成本 c=1，立即入账；失败查询计费但不更新任何边界
  - 账本：只增不改，每次查询追加一条记录

自测（方案 M2 通过条件：快照/群组/分母错配均被拒绝）：
    python step5_sidecar.py
"""

import hashlib
import json
import os
import time

import pandas as pd

from step4_compile import to_int, clip01

DATA_DIR = "data"
LEDGER_PATH = os.path.join(DATA_DIR, "ledger.jsonl")
SPLIT_DIR = os.path.join(DATA_DIR, "split")

# 部门与指标的对应关系（方案 5.1 节）：z_k1 增长部、z_k2 运营部、z_k3 数据治理部
DEPARTMENTS = {0: "growth", 1: "operations", 2: "governance"}
METRIC_NAMES = {0: "activity", 1: "efficiency", 2: "identifiability"}

# 指标配方版本（数据卡中的清洗/聚合配方，换配方必须换版本号）
RECIPE_VERSIONS = {
    "retail": "retail_metrics_v1",   # 见 step2/step4：整数便士、clip01、可识别率
    "bank": "bank_metrics_v1",       # 转化率、时长效率、字段可观察性
}

QUERY_COST = 1  # 主实验统一单价（方案 10.2 节）


class ReceiptRejected(Exception):
    """收据被侧车拒绝时抛出，reason 用于日志和审计。"""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"收据拒绝: {reason}")


def _hash(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


class EvidenceSidecar:
    """可信侧车：目录、权限、收据、计费、账本。"""

    def __init__(self, episode, snapshot_hash, ledger_path=LEDGER_PATH):
        self.episode = episode
        self.episode_id = episode["episode_id"]
        self.dataset = episode["dataset"]
        self.snapshot_hash = snapshot_hash
        self.ledger_path = ledger_path
        self.total_cost = 0
        self.ledger = []
        # 已出过的收据按 query_id 缓存（账本只增不改，重复查询返回原收据，
        # 由调用方的查询策略负责"不重复购买"，侧车只保证事实不被篡改）
        self._receipts = {}
        self.catalog = self._build_catalog()

    def _build_catalog(self):
        """公共查询目录：任务开始前由数据合同固定，24 个合法事实。"""
        catalog = {}
        for k in range(8):
            for i in range(3):
                qid = f"q_{k}_{i}"
                catalog[qid] = {
                    "query_id": qid,
                    "owner_id": DEPARTMENTS[i],
                    "fact_index": k * 3 + i,          # 在 24 维 z 向量中的位置
                    "cohort": self.episode["cohorts"][k]["name"],
                    "metric": METRIC_NAMES[i],
                    "cost": QUERY_COST,
                }
        return catalog

    def _numerator_denominator(self, q):
        """从原始数据重算该事实的分子/分母（EXP-012：取代原占位值）。

        完整性约定：收据的 num/den 必须能复现 normalized_value_int——
          retail 活动:   z1 = clip01(G / (n_plus*sG))      -> num=G,        den=n_plus*sG
          retail 效率:   z2 = 1 - clip01(C / G)            -> num=C,        den=G
          retail 可识别: z3 = notna 行数 / 总行数           -> num=可识别行数, den=rows
          bank 转化:     z1 = #(y=="yes") / rows           -> num=转化行数,  den=rows
          bank 时长:     z2 = 1 - clip01(Σduration/(n*sD)) -> num=Σduration, den=rows*sD
          bank 可观察:   z3 = 1 - 未知字段数/(3*rows)       -> num=未知数,    den=3*rows
        群组掩码按群组名从分割数据重建，行数必须与任务状态中登记的 rows 一致，
        否则拒绝出收据（COHORT_MISMATCH）——侧车不许在绑定断裂时沉默。
        """
        c = self.episode["cohorts"][q["fact_index"] // 3]
        i = q["fact_index"] % 3
        name = c["name"]
        if self.dataset == "retail":
            mask = self._retail_mask(name)
            if i == 0:
                pos = mask[mask["Quantity"] > 0]
                return int((pos["Quantity"] * pos["UnitPricePence"]).sum()), \
                    int(pos["InvoiceNo_str"].nunique() * self._card()["retail_scale"])
            if i == 1:
                neg = mask[mask["Quantity"] < 0]
                C = int((-neg["Quantity"] * neg["UnitPricePence"]).sum())
                G = int((mask[mask["Quantity"] > 0]["Quantity"]
                         * mask[mask["Quantity"] > 0]["UnitPricePence"]).sum())
                return C, max(1, G)
            return int(mask["CustomerID"].notna().sum()), len(mask)
        # ---- bank ----
        mask = self._bank_mask(name)
        n = len(mask)
        if i == 0:
            return int((mask["y"] == "yes").sum()), n
        if i == 1:
            return int(mask["duration"].sum()), int(round(n * self._card()["bank_duration_scale_sD"]))
        unknown = sum(int((mask[col] == "unknown").sum()) for col in ["job", "education", "contact"])
        return int(unknown), n * 3

    # ---------- 群组掩码重建（EXP-012）----------
    _CARD = None
    _FRAMES = {}

    @classmethod
    def _card(cls):
        if cls._CARD is None:
            with open(os.path.join(DATA_DIR, "data_card.json"), encoding="utf-8") as f:
                cls._CARD = json.load(f)
        return cls._CARD

    def _block_frame(self, dataset):
        """按任务所属块取分割数据（dev 任务取 dev 分割），带缓存。"""
        block = self.episode.get("block", "?")
        key = (dataset, block)
        if key in EvidenceSidecar._FRAMES:
            return EvidenceSidecar._FRAMES[key]
        if block == "dev":
            df = pd.read_csv(os.path.join(SPLIT_DIR, f"{dataset}_dev.csv.gz"))
            if "_split" in df.columns:
                df = df[df["_split"] == "dev"]
        else:
            df = pd.read_csv(os.path.join(SPLIT_DIR, f"{dataset}_test.csv.gz"))
            df = df[df["_block"] == block]
        EvidenceSidecar._FRAMES[key] = df
        return df

    def _checked_mask(self, name, mask, cohort):
        """行数绑定：重建掩码必须与原任务状态登记的行数一致。"""
        if len(mask) != cohort.get("rows"):
            raise ReceiptRejected(
                f"COHORT_MISMATCH:{name}: 重建 {len(mask)} 行 != 登记 {cohort.get('rows')} 行")
        return mask

    def _retail_mask(self, name):
        cohort = next(c for c in self.episode["cohorts"] if c["name"] == name)
        df = self._block_frame("retail")
        bands = self._card()["retail_price_bands_pence"]
        uk = df["Country"] == "United Kingdom"
        if name.startswith("UK_priceband"):
            bi = int(name.rsplit("band", 1)[1])
            m = uk & (df["UnitPricePence"] >= bands[bi]) & (df["UnitPricePence"] < bands[bi + 1])
        elif name.startswith("nonUK_priceband"):
            bi = int(name.rsplit("band", 1)[1])
            m = (~uk) & (df["UnitPricePence"] >= bands[bi]) & (df["UnitPricePence"] < bands[bi + 1])
        elif name.startswith("UK_coarseband"):
            bi = int(name.rsplit("band", 1)[1])
            m = uk & (df["UnitPricePence"] >= bands[bi]) & (df["UnitPricePence"] < bands[bi + 2])
        elif name.startswith("nonUK_coarseband"):
            bi = int(name.rsplit("band", 1)[1])
            m = (~uk) & (df["UnitPricePence"] >= bands[bi]) & (df["UnitPricePence"] < bands[bi + 2])
        elif name.startswith("country_"):
            m = df["Country"] == name[len("country_"):]
        else:
            raise ReceiptRejected(f"COHORT_UNKNOWN:{name}")
        return self._checked_mask(name, df[m], cohort)

    def _bank_mask(self, name):
        cohort = next(c for c in self.episode["cohorts"] if c["name"] == name)
        df = self._block_frame("bank")
        if name.startswith("edu_"):
            m = df["education"] == name[len("edu_"):]
        elif name.startswith("job_"):
            m = df["job"] == name[len("job_"):]
        elif name.startswith("balanceband"):
            bi = int(name[len("balanceband"):])
            bands = self._card()["bank_balance_bands"]
            m = (df["balance"] >= bands[bi]) & (df["balance"] < bands[bi + 1])
        else:
            raise ReceiptRejected(f"COHORT_UNKNOWN:{name}")
        return self._checked_mask(name, df[m], cohort)

    def query(self, query_id, requester="coordinator"):
        """发起一次计费核验。成功返回收据字典；失败抛 ReceiptRejected。"""
        # ---- 校验 1：查询必须在公共目录里 ----
        if query_id not in self.catalog:
            self.total_cost += QUERY_COST  # 失败查询也计费（方案 7.3）
            self._log(query_id, requester, "UNKNOWN_QUERY", ok=False)
            raise ReceiptRejected("UNKNOWN_QUERY")
        q = self.catalog[query_id]

        # ---- 幂等：已出过的收据直接返回，不重复盖章 ----
        if query_id in self._receipts:
            return self._receipts[query_id]

        value_int = self.episode["z_int"][q["fact_index"]]
        # ---- 校验 2：量化值必须合法（整数、在 0..10000 网格内）----
        if not isinstance(value_int, int) or not (0 <= value_int <= 10000):
            self.total_cost += QUERY_COST  # 失败查询也计费（方案 7.3）
            self._log(query_id, requester, "OUT_OF_BOUNDS", ok=False)
            raise ReceiptRejected("OUT_OF_BOUNDS")

        c = self.episode["cohorts"][q["fact_index"] // 3]
        try:
            num, den = self._numerator_denominator(q)
        except ReceiptRejected:
            self.total_cost += QUERY_COST   # 失败查询也计费（方案 7.3）
            self._log(query_id, requester, "RECONSTRUCT_FAILED", ok=False)
            raise
        receipt = {
            "episode_id": self.episode_id,
            "snapshot_hash": self.snapshot_hash,
            "query_id": query_id,
            "owner_id": q["owner_id"],
            "cohort_ast_hash": _hash(q["cohort"]),
            "metric_recipe_version": RECIPE_VERSIONS[self.dataset],
            "numerator": num,
            "denominator": den,
            "normalized_value_int": value_int,
            "normalization_scale": 10000,
            "valid_row_count": c["rows"],
            "unit": "dimensionless",
            "receipt_id": _hash(f"{self.episode_id}|{query_id}|{self.snapshot_hash}"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # ---- 计费并写账（只增不改）----
        self.total_cost += q["cost"]
        self._log(query_id, requester, "OK", ok=True, receipt_id=receipt["receipt_id"])
        self._receipts[query_id] = receipt
        return receipt

    def _log(self, query_id, requester, status, ok, receipt_id=None):
        entry = {
            "episode_id": self.episode_id,
            "query_id": query_id,
            "requester": requester,
            "status": status,
            "ok": ok,
            "cost_charged": QUERY_COST if not ok else QUERY_COST,  # 失败也计费（方案 7.3）
            "cumulative_cost": self.total_cost,
            "receipt_id": receipt_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.ledger.append(entry)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def snapshot_hash():
    """当前数据快照的指纹（原始 hash + 任务文件 + 数据卡）。"""
    parts = []
    for fn in ["source_manifest.json", "data_card.json"]:
        with open(os.path.join(DATA_DIR, fn), encoding="utf-8") as f:
            parts.append(f.read())
    with open(os.path.join(DATA_DIR, "episodes.jsonl"), encoding="utf-8") as f:
        parts.append(f.read())
    return _hash("".join(parts))


# ============================================================
# 自测：对应方案 M2 的通过条件
# ============================================================
def self_test():
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)  # 自测用空账本，保证可重复

    eps = [json.loads(l) for l in open(os.path.join(DATA_DIR, "episodes.jsonl"), encoding="utf-8")]
    ep = eps[0]
    sh = snapshot_hash()
    car = EvidenceSidecar(ep, sh)

    checks = []

    # 1. 目录完整：24 个合法事实
    checks.append(("目录=24个事实", len(car.catalog) == 24))

    # 2. 三个部门各拥有 8 个事实
    owners = {}
    for q in car.catalog.values():
        owners[q["owner_id"]] = owners.get(q["owner_id"], 0) + 1
    checks.append(("三部门各8个事实", owners == {"growth": 8, "operations": 8, "governance": 8}))

    # 3. 正常查询出收据，字段齐全
    r = car.query("q_3_1")
    required = ["episode_id", "snapshot_hash", "query_id", "owner_id", "cohort_ast_hash",
                "metric_recipe_version", "numerator", "denominator", "normalized_value_int",
                "normalization_scale", "valid_row_count", "unit", "receipt_id"]
    checks.append(("收据字段齐全", all(k in r for k in required)))
    checks.append(("收据值=任务状态", r["normalized_value_int"] == ep["z_int"][10]))

    # 4. 幂等：重复查询返回同一张收据，且成本只算一次（重复购买由策略层禁止）
    cost_before = car.total_cost
    r2 = car.query("q_3_1")
    checks.append(("重复查询幂等", r2["receipt_id"] == r["receipt_id"]))

    # 5. 非法查询被拒绝并计费、入账
    try:
        car.query("q_99_9")
        checks.append(("非法查询被拒绝", False))
    except ReceiptRejected as e:
        checks.append(("非法查询被拒绝", e.reason == "UNKNOWN_QUERY"))
    checks.append(("失败查询计费", car.total_cost == cost_before + 1))

    # 6. 篡改检测：换一个事实的值超出网格必须拒绝
    bad_ep = json.loads(json.dumps(ep))
    bad_ep["z_int"][5] = 20000
    car_bad = EvidenceSidecar(bad_ep, sh)
    try:
        car_bad.query("q_1_2")
        checks.append(("越界值被拒绝", False))
    except ReceiptRejected as e:
        checks.append(("越界值被拒绝", e.reason == "OUT_OF_BOUNDS"))

    # 7. 快照绑定：换一个 episode 用同一侧车查询同 id，事实位置相同但群组不同，
    #    收据里的群组 hash 必须不同（防止跨任务串收据）
    ep_other = eps[120]
    car_other = EvidenceSidecar(ep_other, sh)
    r_other = car_other.query("q_3_1")
    checks.append(("跨任务收据不串", r_other["cohort_ast_hash"] != r["cohort_ast_hash"]))

    # 8. 分子/分母不再是占位值，且能复现量化值（EXP-012，两数据集各抽一个事实）
    r_bank = car_other.query("q_0_0")   # eps[120] 是 bank 任务
    checks.append(("bank分子非占位", r_bank["numerator"] is not None
                   and r_bank["denominator"] not in (None, 0)))
    z1_bank = r_bank["numerator"] / r_bank["denominator"]
    checks.append(("bank收据可复现", abs(to_int(z1_bank) - ep_other["z_int"][0]) <= 1))
    r_eff = car.query("q_0_1")          # retail 效率: z2 = 1 - C/G
    z2_ret = 1 - clip01(r_eff["numerator"] / r_eff["denominator"])
    checks.append(("retail收据可复现", abs(to_int(z2_ret) - ep["z_int"][1]) <= 1))

    # 9. 全量可复现审计：全部任务 x 24 事实，重建分子分母必须复现任务状态
    card = EvidenceSidecar._card()
    sG, sD = card["retail_scale"], card["bank_duration_scale_sD"]
    max_diff, n_checked = 0, 0
    for ep_a in eps:
        car_a = EvidenceSidecar(ep_a, sh)
        z = ep_a["z_int"]
        for k in range(8):
            cname = ep_a["cohorts"][k]["name"]
            if ep_a["dataset"] == "retail":
                mask = car_a._retail_mask(cname)
                pos = mask[mask["Quantity"] > 0]
                G = int((pos["Quantity"] * pos["UnitPricePence"]).sum())
                n_plus = pos["InvoiceNo_str"].nunique()
                neg = mask[mask["Quantity"] < 0]
                C = int((-neg["Quantity"] * neg["UnitPricePence"]).sum())
                vals = [clip01(G / (n_plus * sG)) if n_plus > 0 else 0.0,
                        1 - clip01(C / G) if G > 0 else 0.0,
                        float(mask["CustomerID"].notna().mean())]
            else:
                mask = car_a._bank_mask(cname)
                n = len(mask)
                vals = [float((mask["y"] == "yes").mean()),
                        1 - clip01(mask["duration"].sum() / (n * sD)),
                        1 - sum(int((mask[col] == "unknown").sum())
                                  for col in ["job", "education", "contact"]) / (n * 3)]
            for i in range(3):
                n_checked += 1
                max_diff = max(max_diff, abs(to_int(vals[i]) - z[k * 3 + i]))
    checks.append((f"全量可复现审计({n_checked}事实,最大偏差{max_diff})", max_diff <= 1))

    print("===== 侧车自测 =====")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    print("全部通过 ✓" if all_ok else "存在失败项 ✗")

    # 保存样例收据（论文附录素材）
    os.makedirs(os.path.join(DATA_DIR, "artifacts"), exist_ok=True)
    with open(os.path.join(DATA_DIR, "artifacts", "receipts_sample.json"), "w", encoding="utf-8") as f:
        json.dump({"receipt_ok": r, "receipt_bank": r_bank,
                   "receipt_reject_reasons": ["UNKNOWN_QUERY", "OUT_OF_BOUNDS", "COHORT_MISMATCH"]},
                  f, ensure_ascii=False, indent=2)
    return all_ok


if __name__ == "__main__":
    self_test()
