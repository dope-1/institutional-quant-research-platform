from __future__ import annotations


def one_way_cost_rate(transaction_cost_bps: float, slippage_bps: float) -> float:
    return (transaction_cost_bps + slippage_bps) / 10_000.0
