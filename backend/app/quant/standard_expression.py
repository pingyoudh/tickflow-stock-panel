"""Standard expression factor parser, evaluator and CSV importer.

The expression language is a small function-style DSL, for example:
Div(TsDelta($high, 1), Add(Abs(TsDelta($high, 1)), 1e-6)).
It is parsed with a recursive descent parser and compiled to Polars
expressions; no eval/exec is used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from app.quant.models import FactorDefinition

LIBRARY_NAME = "标准表达式因子库"
ORIGIN = "standard_expression"
SEMANTIC_VERSION = "standard-expression-v1"

DEFAULT_LIBRARY_DIR = Path(r"C:\Users\dzzz\OneDrive\Desktop\AAA")


class ExpressionSyntaxError(ValueError):
    pass


class StandardExpressionParser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> dict[str, Any]:
        if not self.text:
            raise ExpressionSyntaxError("表达式为空")
        node = self._expr()
        self._skip_ws()
        if self.pos != len(self.text):
            raise ExpressionSyntaxError(f"表达式尾部存在无法解析内容: {self.text[self.pos:self.pos + 20]}")
        return node

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _expr(self) -> dict[str, Any]:
        self._skip_ws()
        char = self._peek()
        if char == "$":
            return self._field()
        if char.isdigit() or char in "+-.":
            return self._number()
        if char.isalpha() or char == "_":
            return self._call()
        raise ExpressionSyntaxError(f"无法解析表达式位置 {self.pos}: {self.text[self.pos:self.pos + 20]}")

    def _field(self) -> dict[str, Any]:
        self.pos += 1
        start = self.pos
        while self.pos < len(self.text) and re.match(r"[A-Za-z0-9_]", self.text[self.pos]):
            self.pos += 1
        if self.pos == start:
            raise ExpressionSyntaxError("$ 后缺少字段名")
        return {"op": "field", "name": self.text[start:self.pos]}

    def _number(self) -> dict[str, Any]:
        start = self.pos
        if self._peek() in "+-":
            self.pos += 1
        has_digit = False
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
            has_digit = True
        if self._peek() == ".":
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
                has_digit = True
        if self._peek() in "eE":
            self.pos += 1
            if self._peek() in "+-":
                self.pos += 1
            exp_start = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            if self.pos == exp_start:
                raise ExpressionSyntaxError("科学计数法指数缺少数字")
        if not has_digit:
            raise ExpressionSyntaxError(f"非法数字: {self.text[start:self.pos + 10]}")
        value = float(self.text[start:self.pos])
        if value.is_integer():
            value = int(value)
        return {"op": "literal", "value": value}

    def _call(self) -> dict[str, Any]:
        start = self.pos
        while self.pos < len(self.text) and re.match(r"[A-Za-z0-9_]", self.text[self.pos]):
            self.pos += 1
        name = self.text[start:self.pos]
        self._skip_ws()
        if self._peek() != "(":
            raise ExpressionSyntaxError(f"算子 {name} 缺少 '('")
        self.pos += 1
        args: list[dict[str, Any]] = []
        self._skip_ws()
        if self._peek() == ")":
            self.pos += 1
            return {"op": "call", "name": name, "args": args}
        while True:
            args.append(self._expr())
            self._skip_ws()
            char = self._peek()
            if char == ",":
                self.pos += 1
                continue
            if char == ")":
                self.pos += 1
                return {"op": "call", "name": name, "args": args}
            raise ExpressionSyntaxError(f"算子 {name} 参数缺少 ',' 或 ')'")


def parse_standard_expression(text: str) -> dict[str, Any]:
    return StandardExpressionParser(text).parse()


def _walk(node: dict[str, Any], fields: set[str], operators: set[str]) -> None:
    if node.get("op") == "field":
        fields.add(str(node.get("name", "")))
        return
    if node.get("op") == "call":
        operators.add(str(node.get("name", "")))
        for arg in node.get("args", []):
            _walk(arg, fields, operators)


FIELD_REQUIREMENTS: dict[str, list[str]] = {
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close"],
    "volume": ["volume"],
    "amount": ["amount"],
    "turn": ["turnover_rate"],
    "vwap": ["amount", "volume"],
    "returns": ["close"],
    "pctchange": ["change_pct", "close"],
    "change": ["change_amount", "close"],
}

ETF_SAFE_RAW_FIELDS = {
    "open", "high", "low", "close", "volume", "amount",
    "vwap", "returns", "pctchange", "change",
}


def standard_expression_asset_types(raw_fields: list[str]) -> list[str]:
    return (
        ["stock", "etf"]
        if set(raw_fields).issubset(ETF_SAFE_RAW_FIELDS)
        else ["stock"]
    )


SUPPORTED_OPERATORS = {
    "Abs", "Add", "And", "ATR", "Atan2", "Div", "Eq", "Exp", "GetLess",
    "Greater", "GreaterEqual", "IfElse", "Inv", "KDJ_D", "KDJ_K", "Less",
    "LessEqual", "Log", "MACDHist", "MACDLine", "MACDSignal", "Max2", "Min2",
    "Mul", "Ne", "Neg", "OBV", "Or", "Pow", "RSI", "Rank", "Ref", "SLog1p",
    "Scale", "Sign", "SignedPower", "Sin", "Slope", "Sqrt", "Square",
    "StochRSI", "Sub", "Tanh", "TsArgMax", "TsArgMin", "TsCorr", "TsCount",
    "TsCov", "TsDelta", "TsDiv", "TsEMA", "TsKurt", "TsMad", "TsMax",
    "TsMean", "TsMed", "TsMin", "TsPctChange", "TsQuantile", "TsRank",
    "TsRatio", "TsSkew", "TsStd", "TsSum", "TsVar", "TsWMA", "WilliamsR",
    "ZScore",
}


UNSUPPORTED_OPERATORS = {
    "RecursiveEMA", "Resi", "Rsquare", "TsDecay", "TsMaxDiff", "TsMinDiff",
    "TsProduct",
}


@dataclass
class ExpressionAnalysis:
    ast: dict[str, Any]
    fields: list[str]
    operators: list[str]
    inputs: list[str]
    warmup: int
    unsupported_fields: list[str] = field(default_factory=list)
    unsupported_operators: list[str] = field(default_factory=list)
    syntax_error: str = ""

    @property
    def ready(self) -> bool:
        return not self.syntax_error and not self.unsupported_fields and not self.unsupported_operators

    @property
    def blocked_reason(self) -> str:
        if self.syntax_error:
            return f"syntax_error: {self.syntax_error}"
        parts = []
        if self.unsupported_fields:
            parts.append("missing_field: " + ", ".join(self.unsupported_fields))
        if self.unsupported_operators:
            parts.append("unsupported_operator: " + ", ".join(self.unsupported_operators))
        return "; ".join(parts)


def analyze_standard_expression(expression: str) -> ExpressionAnalysis:
    try:
        ast = parse_standard_expression(expression)
    except ExpressionSyntaxError as exc:
        return ExpressionAnalysis(
            ast={}, fields=[], operators=[], inputs=[], warmup=0, syntax_error=str(exc)
        )
    fields: set[str] = set()
    operators: set[str] = set()
    _walk(ast, fields, operators)
    unsupported_fields = sorted(name for name in fields if name not in FIELD_REQUIREMENTS)
    unsupported_operators = sorted(
        name for name in operators if name not in SUPPORTED_OPERATORS
    )
    inputs = sorted({
        requirement
        for name in fields
        for requirement in FIELD_REQUIREMENTS.get(name, [])
    })
    return ExpressionAnalysis(
        ast=ast,
        fields=sorted(fields),
        operators=sorted(operators),
        inputs=inputs,
        warmup=_max_window(ast),
        unsupported_fields=unsupported_fields,
        unsupported_operators=unsupported_operators,
    )


def _max_window(node: dict[str, Any]) -> int:
    if node.get("op") != "call":
        return 0
    values = [_max_window(arg) for arg in node.get("args", [])]
    name = str(node.get("name", ""))
    args = node.get("args", [])
    if name.startswith("Ts") or name in {
        "RSI", "ATR", "KDJ_K", "KDJ_D", "WilliamsR", "MACDLine",
        "MACDSignal", "MACDHist", "StochRSI", "Ref", "Slope",
    }:
        for arg in reversed(args[1:]):
            if arg.get("op") == "literal":
                try:
                    values.append(int(arg["value"]))
                    break
                except (TypeError, ValueError):
                    pass
    return max(values, default=0)


def _window(node: dict[str, Any], *, default: int = 1, max_value: int = 5000) -> int:
    if node.get("op") != "literal":
        raise ValueError("窗口参数必须是数字常量")
    value = int(node.get("value"))
    if value < 1 or value > max_value:
        raise ValueError("窗口参数必须在 1~5000 之间")
    return value


def _safe_div(left: pl.Expr | float, right: pl.Expr | float) -> pl.Expr:
    if not isinstance(left, pl.Expr):
        left = pl.lit(left)
    if not isinstance(right, pl.Expr):
        right = pl.lit(right)
    return pl.when(right.abs() > 1e-12).then(left / right).otherwise(None)


def _safe_log(value: pl.Expr) -> pl.Expr:
    return pl.when(value > 0).then(value.log()).otherwise(None)


def _rolling(value: pl.Expr, name: str, window: int) -> pl.Expr:
    method = {
        "mean": value.rolling_mean,
        "sum": value.rolling_sum,
        "std": value.rolling_std,
        "var": value.rolling_var,
        "min": value.rolling_min,
        "max": value.rolling_max,
        "median": value.rolling_median,
        "skew": value.rolling_skew,
        "kurtosis": value.rolling_kurtosis,
    }[name]
    try:
        return method(window_size=window, min_samples=window).over("symbol")
    except TypeError:
        return method(window_size=window).over("symbol")


class StandardExpressionCompiler:
    def __init__(self, available: set[str]) -> None:
        self.available = available

    def compile(self, node: dict[str, Any]) -> pl.Expr:
        op = node.get("op")
        if op == "literal":
            return pl.lit(node.get("value"))
        if op == "field":
            return self._field(str(node.get("name", "")))
        if op != "call":
            raise ValueError(f"不支持的表达式节点: {op}")
        name = str(node.get("name", ""))
        args = node.get("args", [])
        if name not in SUPPORTED_OPERATORS:
            raise ValueError(f"不支持的标准表达式算子: {name}")
        return self._call(name, args)

    def _field(self, name: str) -> pl.Expr:
        if name.startswith("__std_expr_") and name in self.available:
            return pl.col(name)
        if name in {"open", "high", "low", "close", "volume", "amount"}:
            self._require(name)
            return pl.col(name).cast(pl.Float64, strict=False)
        if name == "turn":
            self._require("turnover_rate")
            return pl.col("turnover_rate").cast(pl.Float64, strict=False)
        if name == "vwap":
            self._require("amount")
            self._require("volume")
            return _safe_div(
                pl.col("amount").cast(pl.Float64, strict=False),
                pl.col("volume").cast(pl.Float64, strict=False),
            )
        if name == "returns":
            self._require("close")
            return pl.col("close").cast(pl.Float64, strict=False).pct_change(1).over("symbol")
        if name == "pctchange":
            if "change_pct" in self.available:
                return pl.col("change_pct").cast(pl.Float64, strict=False)
            self._require("close")
            return pl.col("close").cast(pl.Float64, strict=False).pct_change(1).over("symbol")
        if name == "change":
            if "change_amount" in self.available:
                return pl.col("change_amount").cast(pl.Float64, strict=False)
            self._require("close")
            return pl.col("close").cast(pl.Float64, strict=False).diff(1).over("symbol")
        raise ValueError(f"标准表达式字段暂不可计算: ${name}")

    def _require(self, name: str) -> None:
        if name not in self.available:
            raise ValueError(f"数据面板缺少标准表达式字段: {name}")

    def _call(self, name: str, args: list[dict[str, Any]]) -> pl.Expr:
        if name in {"Add", "Sub", "Mul", "Div", "Pow", "Max2", "Min2", "Greater",
                    "Less", "GreaterEqual", "LessEqual", "Eq", "Ne", "And", "Or",
                    "Atan2", "TsCorr", "TsCov"}:
            if len(args) < 2:
                raise ValueError(f"{name} 至少需要 2 个参数")
            a = self.compile(args[0])
            b = self.compile(args[1])
            if name == "Add":
                return a + b
            if name == "Sub":
                return a - b
            if name == "Mul":
                return a * b
            if name == "Div":
                return _safe_div(a, b)
            if name == "Pow":
                return a.pow(b)
            if name == "Max2":
                return pl.max_horizontal(a, b)
            if name == "Min2":
                return pl.min_horizontal(a, b)
            if name == "Greater":
                return a > b
            if name == "Less":
                return a < b
            if name == "GreaterEqual":
                return a >= b
            if name == "LessEqual":
                return a <= b
            if name == "Eq":
                return a == b
            if name == "Ne":
                return a != b
            if name == "And":
                return a.cast(pl.Boolean, strict=False) & b.cast(pl.Boolean, strict=False)
            if name == "Or":
                return a.cast(pl.Boolean, strict=False) | b.cast(pl.Boolean, strict=False)
            if name == "Atan2":
                return a.arctan2(b)
            window = _window(args[2] if len(args) > 2 else {"op": "literal", "value": 20})
            if name == "TsCorr":
                return pl.rolling_corr(a, b, window_size=window, min_samples=window).over("symbol")
            return pl.rolling_cov(a, b, window_size=window, min_samples=window).over("symbol")

        if name == "IfElse":
            if len(args) != 3:
                raise ValueError("IfElse 需要 3 个参数")
            return pl.when(self.compile(args[0]).cast(pl.Boolean, strict=False)).then(
                self.compile(args[1])
            ).otherwise(self.compile(args[2]))

        if name in {"Abs", "Log", "SLog1p", "Sign", "Inv", "Neg", "Sqrt", "Square",
                    "Exp", "Tanh", "Sin", "SignedPower", "Rank", "ZScore", "Scale"}:
            if len(args) < 1:
                raise ValueError(f"{name} 需要 1 个参数")
            value = self.compile(args[0])
            if name == "Abs":
                return value.abs()
            if name == "Log":
                return _safe_log(value)
            if name == "SLog1p":
                return value.sign() * (value.abs() + 1).log()
            if name == "Sign":
                return value.sign()
            if name == "Inv":
                return _safe_div(pl.lit(1.0), value)
            if name == "Neg":
                return -value
            if name == "Sqrt":
                return pl.when(value >= 0).then(value.sqrt()).otherwise(None)
            if name == "Square":
                return value * value
            if name == "Exp":
                return value.exp()
            if name == "Tanh":
                return value.tanh()
            if name == "Sin":
                return value.sin()
            if name == "SignedPower":
                exponent = self.compile(args[1]) if len(args) > 1 else pl.lit(1.0)
                return value.sign() * value.abs().pow(exponent)
            if name == "Rank":
                return value.rank(method="average").over("date") / pl.len().over("date")
            if name == "ZScore":
                return _safe_div(value - value.mean().over("date"), value.std().over("date"))
            return _safe_div(value, value.abs().sum().over("date"))

        if name in {"Ref", "TsDelta", "TsRatio", "TsDiv", "TsPctChange", "TsMean",
                    "TsSum", "TsStd", "TsVar", "TsMin", "TsMax", "TsMed", "TsMad",
                    "TsSkew", "TsKurt", "TsRank", "TsEMA", "TsWMA", "TsCount",
                    "TsQuantile", "TsArgMin", "TsArgMax"}:
            if len(args) < 2:
                raise ValueError(f"{name} 需要数值和窗口参数")
            value = self.compile(args[0])
            window = _window(args[1])
            if name == "Ref":
                return value.shift(window).over("symbol")
            if name == "TsDelta":
                return value.diff(window).over("symbol")
            if name in {"TsRatio", "TsDiv"}:
                return _safe_div(value, value.shift(window).over("symbol"))
            if name == "TsPctChange":
                return value.pct_change(window).over("symbol")
            if name == "TsMean":
                return _rolling(value, "mean", window)
            if name == "TsSum":
                return _rolling(value, "sum", window)
            if name == "TsStd":
                return _rolling(value, "std", window)
            if name == "TsVar":
                return _rolling(value, "var", window)
            if name == "TsMin":
                return _rolling(value, "min", window)
            if name == "TsMax":
                return _rolling(value, "max", window)
            if name == "TsMed":
                return _rolling(value, "median", window)
            if name == "TsMad":
                median = _rolling(value, "median", window)
                return _rolling((value - median).abs(), "median", window)
            if name == "TsSkew":
                return _rolling(value, "skew", window)
            if name == "TsKurt":
                return _rolling(value, "kurtosis", window)
            if name == "TsRank":
                return value.rolling_rank(window_size=window, method="average", min_samples=window).over("symbol") / window
            if name == "TsEMA":
                return value.ewm_mean(span=window, adjust=False).over("symbol")
            if name == "TsWMA":
                weights = [float(index + 1) for index in range(window)]
                weighted = (
                    value.cast(pl.Float64, strict=False)
                    .fill_null(float("nan"))
                    .rolling_mean(
                        window_size=window,
                        weights=weights,
                        min_samples=window,
                    )
                    .over("symbol")
                )
                return weighted.fill_nan(None)
            if name == "TsCount":
                return value.cast(pl.Int8, strict=False).rolling_sum(window_size=window, min_samples=window).over("symbol")
            if name == "TsQuantile":
                q = float(args[2].get("value", 0.5)) if len(args) > 2 and args[2].get("op") == "literal" else 0.5
                return value.rolling_quantile(q, window_size=window, min_samples=window).over("symbol")
            if name == "TsArgMin":
                return value.rolling_map(_arg_min, window_size=window, min_samples=window).over("symbol")
            return value.rolling_map(_arg_max, window_size=window, min_samples=window).over("symbol")

        if name == "GetLess":
            return pl.when(self.compile(args[0]) < self.compile(args[1])).then(self.compile(args[0])).otherwise(None)
        if name == "RSI":
            value = self.compile(args[0])
            window = _window(args[1])
            diff = value.diff(1).over("symbol")
            gain = pl.when(diff > 0).then(diff).otherwise(0.0)
            loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
            avg_gain = _rolling(gain, "mean", window)
            avg_loss = _rolling(loss, "mean", window)
            return (
                pl.when((avg_loss.abs() <= 1e-12) & (avg_gain.abs() <= 1e-12)).then(50.0)
                .when(avg_loss.abs() <= 1e-12).then(100.0)
                .otherwise(100.0 - _safe_div(100.0, 1.0 + _safe_div(avg_gain, avg_loss)))
            )
        if name == "ATR":
            high, low, close = self.compile(args[0]), self.compile(args[1]), self.compile(args[2])
            window = _window(args[3])
            prev_close = close.shift(1).over("symbol")
            tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
            return _rolling(tr, "mean", window)
        if name in {"KDJ_K", "KDJ_D"}:
            high, low, close = self.compile(args[0]), self.compile(args[1]), self.compile(args[2])
            window = _window(args[3])
            low_min = _rolling(low, "min", window)
            high_max = _rolling(high, "max", window)
            rsv = _safe_div(close - low_min, high_max - low_min) * 100.0
            k = rsv.ewm_mean(alpha=1 / 3, adjust=False).over("symbol")
            if name == "KDJ_K":
                return k
            smooth = _window(args[4]) if len(args) > 4 else 3
            return k.ewm_mean(alpha=1 / smooth, adjust=False).over("symbol")
        if name in {"MACDLine", "MACDSignal", "MACDHist"}:
            value = self.compile(args[0])
            fast = _window(args[1])
            slow = _window(args[2])
            line = value.ewm_mean(span=fast, adjust=False).over("symbol") - value.ewm_mean(span=slow, adjust=False).over("symbol")
            if name == "MACDLine":
                return line
            signal_window = _window(args[3]) if len(args) > 3 else 9
            signal = line.ewm_mean(span=signal_window, adjust=False).over("symbol")
            if name == "MACDSignal":
                return signal
            return line - signal
        if name == "WilliamsR":
            high, low, close = self.compile(args[0]), self.compile(args[1]), self.compile(args[2])
            window = _window(args[3])
            highest = _rolling(high, "max", window)
            lowest = _rolling(low, "min", window)
            return _safe_div(highest - close, highest - lowest) * -100.0
        if name == "StochRSI":
            rsi = self._call("RSI", args)
            window = _window(args[1])
            return _safe_div(rsi - _rolling(rsi, "min", window), _rolling(rsi, "max", window) - _rolling(rsi, "min", window))
        if name == "OBV":
            close, volume = self.compile(args[0]), self.compile(args[1])
            diff = close.diff(1).over("symbol")
            signed = pl.when(diff > 0).then(volume).when(diff < 0).then(-volume).otherwise(0.0)
            return signed.cum_sum().over("symbol")
        if name == "Slope":
            value = self.compile(args[0])
            window = _window(args[1])
            weights = [float(index) for index in range(window)]
            x_mean = (window - 1) / 2
            denom = sum((index - x_mean) ** 2 for index in range(window))
            weighted_sum = value.rolling_sum(
                window_size=window, weights=weights, min_samples=window
            ).over("symbol")
            y_sum = value.rolling_sum(window_size=window, min_samples=window).over("symbol")
            return _safe_div(weighted_sum - x_mean * y_sum, denom)

        raise ValueError(f"不支持的标准表达式算子: {name}")


def _series_values(series) -> list[float]:
    return [float(value) for value in series if value is not None and math.isfinite(float(value))]


def _last_rank_pct(series) -> float:
    values = _series_values(series)
    if len(values) < len(series):
        return math.nan
    last = values[-1]
    rank = 1 + sum(value < last for value in values)
    ties = sum(value == last for value in values)
    return (rank + (ties - 1) / 2) / len(values)


def _wma(series) -> float:
    values = _series_values(series)
    if len(values) < len(series):
        return math.nan
    weights = list(range(1, len(values) + 1))
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _arg_min(series) -> float:
    values = _series_values(series)
    return float(values.index(min(values))) if values else math.nan


def _arg_max(series) -> float:
    values = _series_values(series)
    return float(values.index(max(values))) if values else math.nan


def _slope(series) -> float:
    values = _series_values(series)
    if len(values) < 2:
        return math.nan
    n = len(values)
    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return math.nan
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def compile_standard_expression(node: dict[str, Any], available: set[str]) -> pl.Expr:
    return StandardExpressionCompiler(available).compile(node)


class StandardExpressionPlan:
    def __init__(self, available: set[str]) -> None:
        self.available = set(available)
        self.steps: list[tuple[str, dict[str, Any]]] = []
        self._index = 0

    def build(self, node: dict[str, Any]) -> dict[str, Any]:
        op = node.get("op")
        if op in {"field", "literal"}:
            return dict(node)
        if op != "call":
            raise ValueError(f"不支持的表达式节点: {op}")
        staged_args: list[dict[str, Any]] = []
        for arg in node.get("args", []):
            staged = self.build(arg)
            if staged.get("op") == "call":
                name = self._next_temp()
                self.steps.append((name, staged))
                self.available.add(name)
                staged_args.append({"op": "field", "name": name})
            else:
                staged_args.append(staged)
        return {"op": "call", "name": node.get("name"), "args": staged_args}

    def _next_temp(self) -> str:
        while True:
            self._index += 1
            name = f"__std_expr_{self._index}"
            if name not in self.available:
                return name


def evaluate_standard_expression(
    frame: pl.DataFrame,
    node: dict[str, Any],
    *,
    output_name: str = "value",
) -> pl.DataFrame:
    plan = StandardExpressionPlan(set(frame.columns))
    final_node = plan.build(node)
    out = frame
    for name, step in plan.steps:
        expr = compile_standard_expression(step, plan.available)
        out = out.with_columns(expr.alias(name))
    final_expr = compile_standard_expression(final_node, plan.available)
    out = out.with_columns(final_expr.cast(pl.Float64, strict=False).alias(output_name))
    temp_columns = [name for name, _ in plan.steps if name in out.columns]
    return out.drop(temp_columns) if temp_columns else out


def standard_expression_version(expression: str, ast: dict[str, Any]) -> str:
    payload = {
        "semantic_version": SEMANTIC_VERSION,
        "expression": expression,
        "ast": ast,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class FactorCsvRecord:
    expression: str
    factor_type: str = ""
    description: str = ""
    source: str = ""
    source_file: str = ""
    source_row: str = ""
    admission_status: str = "unscreened"


def import_standard_expression_library(root: Path, *, dry_run: bool = True) -> dict[str, Any]:
    records, source_rows = _load_records(root)
    definitions = [_record_to_definition(record) for record in records.values()]
    stats = _import_stats(definitions, source_rows)
    stats["factors"] = definitions
    if dry_run:
        stats["preview"] = [item.model_dump(mode="json") for item in definitions[:20]]
    return stats


def _load_records(root: Path) -> tuple[dict[str, FactorCsvRecord], int]:
    files = [
        ("候选因子库.csv", "候选因子库", "unscreened"),
        (str(Path("因子筛选") / "因子筛选" / "基础因子库.csv"), "基础因子库", "admitted"),
        (str(Path("因子筛选") / "因子筛选" / "机器学习因子库.csv"), "机器学习因子库", "admitted"),
        (str(Path("因子筛选") / "因子筛选" / "因子筛选明细.csv"), "因子筛选明细", "unscreened"),
        (str(Path("因子筛选") / "因子筛选" / "暂未进入因子库.csv"), "暂未进入因子库", "rejected"),
    ]
    records: dict[str, FactorCsvRecord] = {}
    source_rows = 0
    for relative, source, default_status in files:
        path = root / relative
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_no, row in enumerate(csv.DictReader(handle), start=2):
                expression = str(row.get("算子表达式", "") or "").strip()
                if not expression:
                    continue
                source_rows += 1
                status = _normalize_admission_status(row.get("入库状态") or default_status)
                record = records.get(expression)
                if record is None:
                    record = FactorCsvRecord(expression=expression)
                    records[expression] = record
                record.factor_type = record.factor_type or str(row.get("类型", "") or "未分类").strip()
                record.description = record.description or str(row.get("因子解释", "") or "").strip()
                record.source = _merge_source(record.source, source)
                record.source_file = record.source_file or str(path)
                record.source_row = record.source_row or str(row.get("序号") or row.get("原序号") or row_no)
                record.admission_status = _merge_status(record.admission_status, status)
    return dict(sorted(records.items(), key=lambda item: item[0])), source_rows


def _normalize_admission_status(value: str) -> str:
    if value in {"入库", "admitted"}:
        return "admitted"
    if value in {"不入库", "rejected"}:
        return "rejected"
    return "unscreened"


def _merge_status(current: str, incoming: str) -> str:
    if "admitted" in {current, incoming}:
        return "admitted"
    if "rejected" in {current, incoming}:
        return "rejected"
    return "unscreened"


def _merge_source(current: str, incoming: str) -> str:
    values = [part for part in current.split(",") if part] if current else []
    if incoming not in values:
        values.append(incoming)
    return ",".join(values)


def _record_to_definition(record: FactorCsvRecord) -> FactorDefinition:
    analysis = analyze_standard_expression(record.expression)
    factor_id = "expr_factor_" + hashlib.sha256(record.expression.encode("utf-8")).hexdigest()[:16]
    name = _readable_name(record)
    compute_status = "ready" if analysis.ready else "blocked"
    enabled = record.admission_status == "admitted" and analysis.ready
    tags = [part for part in [record.factor_type, record.admission_status, compute_status] if part]
    return FactorDefinition(
        id=factor_id,
        name=name,
        description=_truncate(record.description, 500),
        family=_truncate(record.factor_type or "标准表达式", 32),
        version=standard_expression_version(record.expression, analysis.ast),
        authoring_type="declarative",
        asset_types=standard_expression_asset_types(analysis.fields),
        inputs=analysis.inputs,
        warmup=min(max(analysis.warmup * 2, 20), 5000),
        expression=analysis.ast,
        trusted=True,
        readonly=False,
        enabled=enabled,
        origin=ORIGIN,
        library_name=LIBRARY_NAME,
        admission_status=record.admission_status,
        compute_status=compute_status,
        blocked_reason=analysis.blocked_reason,
        source_expression=record.expression,
        source_file=record.source_file,
        source_row=record.source_row,
        tags=tags,
        operators=analysis.operators,
        raw_fields=analysis.fields,
        params={"source": record.source, "semantic_version": SEMANTIC_VERSION},
    )


def _readable_name(record: FactorCsvRecord) -> str:
    expression = record.expression
    factor_type = record.factor_type or "标准表达式"
    desc = re.sub(r"\s+", "", record.description or "")
    for pattern, label in [
        (r"TsMean\(\$close,\s*(\d+)\)", r"\1日收盘均线"),
        (r"TsEMA\(\$close,\s*(\d+)\)", r"\1日收盘EMA"),
        (r"TsPctChange\(\$close,\s*(\d+)\)", r"\1日收盘收益率"),
        (r"TsStd\(\$returns,\s*(\d+)\)", r"\1日收益波动率"),
        (r"RSI\(\$close,\s*(\d+)\)", r"RSI\1"),
    ]:
        match = re.fullmatch(pattern, expression)
        if match:
            return match.expand(label)
    if desc:
        first = re.split(r"[。；;，,]", desc)[0]
        first = re.sub(r"^(计算每只股票|计算|衡量|刻画|识别)", "", first)
        if first:
            return _truncate(first, 30)
    return _truncate(factor_type.replace("-", "") + "因子", 30)


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    return value[:limit] if len(value) > limit else value


def _import_stats(definitions: list[FactorDefinition], source_rows: int) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    compute_counts: dict[str, int] = {}
    missing_fields: dict[str, int] = {}
    unsupported_ops: dict[str, int] = {}
    for item in definitions:
        status_counts[item.admission_status] = status_counts.get(item.admission_status, 0) + 1
        compute_counts[item.compute_status] = compute_counts.get(item.compute_status, 0) + 1
        if item.compute_status != "ready":
            for raw in item.raw_fields:
                if raw not in FIELD_REQUIREMENTS:
                    missing_fields[raw] = missing_fields.get(raw, 0) + 1
            for op in item.operators:
                if op not in SUPPORTED_OPERATORS:
                    unsupported_ops[op] = unsupported_ops.get(op, 0) + 1
    return {
        "library_name": LIBRARY_NAME,
        "origin": ORIGIN,
        "source_root": str(DEFAULT_LIBRARY_DIR),
        "source_rows": source_rows,
        "unique_expressions": len(definitions),
        "admission_status": status_counts,
        "compute_status": compute_counts,
        "enabled": sum(1 for item in definitions if item.enabled),
        "blocked": sum(1 for item in definitions if item.compute_status != "ready"),
        "missing_fields": dict(sorted(missing_fields.items())),
        "unsupported_operators": dict(sorted(unsupported_ops.items())),
    }
