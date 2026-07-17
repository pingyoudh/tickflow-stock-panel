import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Archive, BarChart3, CheckCircle2, Download, Loader2, Play, RefreshCw,
  Rocket, Search, SlidersHorizontal, TriangleAlert,
} from 'lucide-react'
import {
  api, type MLBacktestHolding, type MLBacktestSpec, type MLBacktestTrade,
  type MLDiagnostic, type MLEquityPoint, type MLPredictionRow,
  type QuantModel, type QuantModelDetail,
} from '@/lib/api'

type DetailView = 'validation' | 'backtest' | 'predictions'

const DETAIL_TABS: { id: DetailView; label: string; icon: typeof CheckCircle2 }[] = [
  { id: 'validation', label: 'OOS 验证', icon: CheckCircle2 },
  { id: 'backtest', label: '组合回测', icon: BarChart3 },
  { id: 'predictions', label: '盘后预测', icon: RefreshCw },
]

const inputClass = 'h-8 w-full rounded-input border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent'
const actionClass = 'inline-flex h-8 items-center justify-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-50'
const quietButton = 'inline-flex h-7 items-center justify-center gap-1 rounded-btn border border-border bg-base px-2 text-[11px] text-secondary hover:bg-elevated hover:text-foreground disabled:opacity-40'

const gradeLabel = {
  robust: '稳健', candidate: '候选', weak: '偏弱', invalid: '无效', unverified: '待验证',
} as const

const gradeClass = {
  robust: 'border-bear/30 bg-bear/10 text-bear',
  candidate: 'border-accent/30 bg-accent/10 text-accent',
  weak: 'border-warning/30 bg-warning/10 text-warning',
  invalid: 'border-danger/30 bg-danger/10 text-danger',
  unverified: 'border-border bg-elevated text-muted',
} as const

function pct(value: unknown, digits = 2) {
  const number = Number(value)
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : '--'
}

function num(value: unknown, digits = 2) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : '--'
}

function Grade({ diagnostic }: { diagnostic?: MLDiagnostic }) {
  const grade = diagnostic?.grade ?? 'unverified'
  return <span className={`inline-flex rounded border px-1.5 py-0.5 text-[10px] ${gradeClass[grade]}`}>{gradeLabel[grade]}</span>
}

function DimensionDot({ status }: { status: 'green' | 'yellow' | 'red' }) {
  return <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${status === 'green' ? 'bg-bear' : status === 'yellow' ? 'bg-warning' : 'bg-danger'}`} />
}

function EquityChart({ rows }: { rows: MLEquityPoint[] }) {
  if (rows.length < 2) return <div className="py-14 text-center text-xs text-muted">暂无净值数据</div>
  const series = [
    { key: 'value' as const, color: '#3b82f6' },
    { key: 'index_benchmark' as const, color: '#ef4444' },
    { key: 'universe_benchmark' as const, color: '#22c55e' },
  ]
  const values = series.flatMap(item => rows.map(row => Number(row[item.key])).filter(Number.isFinite))
  const low = Math.min(...values)
  const high = Math.max(...values)
  const range = Math.max(high - low, 1)
  const path = (key: typeof series[number]['key']) => rows.map((row, index) => {
    const x = 12 + index / (rows.length - 1) * 776
    const y = 204 - (Number(row[key]) - low) / range * 184
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return <div className="border-y border-border bg-surface p-3">
    <div className="mb-2 flex flex-wrap gap-4 text-[10px] text-muted"><span className="text-accent">策略净值</span><span className="text-danger">目标指数</span><span className="text-bear">股票池等权</span></div>
    <svg viewBox="0 0 800 220" className="h-52 w-full" role="img" aria-label="OOS净值与双基准">
      {[20, 66, 112, 158, 204].map(y => <line key={y} x1="12" x2="788" y1={y} y2={y} stroke="currentColor" className="text-border" strokeWidth="1" />)}
      {series.map(item => <path key={item.key} d={path(item.key)} fill="none" stroke={item.color} strokeWidth="2" vectorEffect="non-scaling-stroke" />)}
    </svg>
  </div>
}

function ICChart({ rows }: { rows: { date: string; ic: number }[] }) {
  if (rows.length < 2) return <div className="py-10 text-center text-xs text-muted">暂无每日 IC</div>
  const maxAbs = Math.max(0.1, ...rows.map(item => Math.abs(item.ic)))
  const points = rows.map((row, index) => {
    const x = 10 + index / (rows.length - 1) * 780
    const y = 100 - row.ic / maxAbs * 82
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return <svg viewBox="0 0 800 200" className="h-44 w-full" role="img" aria-label="每日Rank IC曲线">
    <line x1="10" x2="790" y1="100" y2="100" stroke="currentColor" className="text-border" />
    <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" vectorEffect="non-scaling-stroke" />
  </svg>
}

function PredictionDistribution({ rows }: { rows: MLPredictionRow[] }) {
  const bins = useMemo(() => {
    const values = rows.map(item => Number(item.prediction)).filter(Number.isFinite)
    if (!values.length) return []
    const low = Math.min(...values)
    const high = Math.max(...values)
    const width = Math.max((high - low) / 12, 1e-12)
    const result = Array.from({ length: 12 }, (_, index) => ({
      from: low + index * width,
      to: index === 11 ? high : low + (index + 1) * width,
      count: 0,
    }))
    values.forEach(value => {
      const index = Math.min(11, Math.max(0, Math.floor((value - low) / width)))
      result[index].count += 1
    })
    return result
  }, [rows])
  const maxCount = Math.max(1, ...bins.map(item => item.count))
  if (!bins.length) return null
  return <div className="border-y border-border bg-surface p-3">
    <div className="mb-3 flex items-center justify-between"><div className="text-xs font-medium">预测值分布</div><div className="text-[10px] text-muted">当前截面 {rows.length} 只</div></div>
    <div className="grid h-28 grid-cols-12 items-end gap-1" role="img" aria-label="预测值分布直方图">{bins.map(item => <div key={item.from} className="group relative flex h-full items-end"><div className="w-full bg-accent/70 transition-colors group-hover:bg-accent" style={{ height: `${Math.max(2, item.count / maxCount * 100)}%` }} /><div className="pointer-events-none absolute bottom-full left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap border border-border bg-base px-1.5 py-1 text-[9px] text-secondary shadow group-hover:block">{num(item.from, 4)} ~ {num(item.to, 4)} · {item.count}</div></div>)}</div>
    <div className="mt-1 flex justify-between font-mono text-[9px] text-muted"><span>{num(bins[0].from, 5)}</span><span>{num(bins[bins.length - 1].to, 5)}</span></div>
  </div>
}

function ValidationView({ detail }: { detail: QuantModelDetail }) {
  const diagnostic = detail.diagnostic
  const result = detail.training_run?.result ?? {}
  const dailyIC = (result.metrics?.daily_ic ?? []) as { date: string; ic: number }[]
  const folds = (result.folds ?? []) as Record<string, any>[]
  const importance = Object.entries(result.feature_importance ?? {}) as [string, { gain: number; split: number }][]
  const maxGain = Math.max(1e-12, ...importance.map(([, item]) => item.gain))
  return <div className="space-y-4">
    <div className="grid gap-px overflow-hidden border-y border-border bg-border md:grid-cols-4">
      {Object.entries(diagnostic.dimensions).map(([key, item]) => <div key={key} className="bg-surface p-3">
        <div className="flex items-center gap-2 text-[11px] text-muted"><DimensionDot status={item.status} />{{ data: '数据质量', statistics: '统计能力', stability: '折间稳定', economics: '经济有效性' }[key]}</div>
        <div className="mt-2 text-xs leading-5 text-secondary">{item.reason}</div>
      </div>)}
    </div>
    {diagnostic.warnings.map(item => <div key={item} className="flex items-start gap-2 border-y border-warning/20 bg-warning/5 px-3 py-2 text-[11px] text-warning"><TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />{item}</div>)}
    <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
      <div className="border-y border-border bg-surface p-3"><div className="text-xs font-medium">每日 Rank IC</div><ICChart rows={dailyIC} /></div>
      <div className="overflow-x-auto border-y border-border bg-surface"><table className="w-full min-w-[500px] text-left text-[11px]"><thead className="bg-elevated/50 text-muted"><tr><th className="px-3 py-2 font-normal">测试折</th><th className="px-3 py-2 font-normal">区间</th><th className="px-3 py-2 font-normal">Rank IC</th><th className="px-3 py-2 font-normal">ICIR</th></tr></thead><tbody>{folds.map(item => <tr key={item.index} className="border-t border-border"><td className="px-3 py-2">Fold {item.index + 1}</td><td className="px-3 py-2 font-mono text-muted">{item.test_start} ~ {item.test_end}</td><td className="px-3 py-2 font-mono">{num(item.metrics?.rank_ic, 4)}</td><td className="px-3 py-2 font-mono">{num(item.metrics?.icir, 2)}</td></tr>)}</tbody></table></div>
    </div>
    <div className="border-y border-border bg-surface p-3"><div className="mb-2 text-xs font-medium">特征重要性</div>{importance.map(([name, item]) => <div key={name} className="grid grid-cols-[120px_1fr_70px] items-center gap-3 border-t border-border py-2 text-[11px]"><span className="truncate font-mono">{name}</span><div className="h-1.5 bg-elevated"><div className="h-full bg-accent" style={{ width: `${item.gain / maxGain * 100}%` }} /></div><span className="text-right font-mono text-muted">{num(item.gain, 6)}</span></div>)}</div>
  </div>
}

function BacktestView({ detail }: { detail: QuantModelDetail }) {
  const queryClient = useQueryClient()
  const horizon = detail.spec.target.horizon
  const [topN, setTopN] = useState(10)
  const [rebalanceDays, setRebalanceDays] = useState<number>(horizon)
  const [weighting, setWeighting] = useState<'equal' | 'score'>('equal')
  const [commission, setCommission] = useState(0.02)
  const [stampTax, setStampTax] = useState(0.05)
  const [slippage, setSlippage] = useState(5)
  const mutation = useMutation({
    mutationFn: (spec: MLBacktestSpec) => api.quantModelBacktest(detail.version, spec),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['quant', 'model', detail.version] })
      void queryClient.invalidateQueries({ queryKey: ['quant', 'experiments'] })
    },
  })
  const active = detail.backtests.find(item => ['queued', 'running', 'cancelling'].includes(item.status))
  const latest = detail.latest_backtest
  const metrics = latest?.result?.metrics ?? {}
  const rows = (latest?.result?.equity_curve ?? []) as MLEquityPoint[]
  const holdings = (latest?.result?.holdings ?? []) as MLBacktestHolding[]
  const trades = (latest?.result?.trades ?? []) as MLBacktestTrade[]
  const latestHoldingDate = holdings.at(-1)?.date
  const latestHoldings = holdings.filter(item => item.date === latestHoldingDate).sort((a, b) => b.weight - a.weight)
  const recentTrades = [...trades].reverse().slice(0, 100)
  const submit = () => mutation.mutate({
    model_version: detail.version, top_n: topN, rebalance_days: rebalanceDays,
    weighting, initial_capital: 1_000_000, commission_pct: commission / 100,
    stamp_tax_pct: stampTax / 100, slippage_bps: slippage,
  })
  return <div className="space-y-4">
    <div className="grid gap-4 border-y border-border bg-surface p-3 xl:grid-cols-[360px_1fr]">
      <div className="grid grid-cols-2 gap-3">
        <label className="text-[11px] text-muted">Top N<input type="number" min={1} max={100} className={`${inputClass} mt-1`} value={topN} onChange={event => setTopN(Number(event.target.value))} /></label>
        <label className="text-[11px] text-muted">调仓交易日<input type="number" min={1} max={252} className={`${inputClass} mt-1`} value={rebalanceDays} onChange={event => setRebalanceDays(Number(event.target.value))} /></label>
        <label className="text-[11px] text-muted">权重<select className={`${inputClass} mt-1`} value={weighting} onChange={event => setWeighting(event.target.value as typeof weighting)}><option value="equal">等权</option><option value="score">分数权重</option></select></label>
        <label className="text-[11px] text-muted">佣金 %<input type="number" step="0.01" className={`${inputClass} mt-1`} value={commission} onChange={event => setCommission(Number(event.target.value))} /></label>
        <label className="text-[11px] text-muted">印花税 %<input type="number" step="0.01" className={`${inputClass} mt-1`} value={stampTax} onChange={event => setStampTax(Number(event.target.value))} /></label>
        <label className="text-[11px] text-muted">滑点 bps<input type="number" step="1" className={`${inputClass} mt-1`} value={slippage} onChange={event => setSlippage(Number(event.target.value))} /></label>
        <button className={`${actionClass} col-span-2`} disabled={mutation.isPending || Boolean(active)} onClick={submit}>{mutation.isPending || active ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}运行严格 OOS 回测</button>
      </div>
      <div className="grid grid-cols-2 gap-px bg-border sm:grid-cols-4">{[
        ['累计收益', pct(metrics.total_return)], ['指数超额', pct(metrics.excess_vs_index)],
        ['股票池超额', pct(metrics.excess_vs_universe)], ['Sharpe', num(metrics.sharpe)],
        ['最大回撤', pct(metrics.max_drawdown)], ['总成本', num(metrics.total_cost, 0)],
        ['交易次数', num(metrics.trade_count, 0)], ['OOS天数', num(metrics.oos_trading_days, 0)],
      ].map(([label, value]) => <div key={label} className="bg-base p-3"><div className="text-[10px] text-muted">{label}</div><div className="mt-1 font-mono text-sm">{value}</div></div>)}</div>
    </div>
    {active && <div className="border-y border-border bg-surface px-3 py-2"><div className="flex justify-between text-[11px] text-muted"><span>{active.message}</span><span>{Math.round(active.progress * 100)}%</span></div><div className="mt-2 h-1 bg-elevated"><div className="h-full bg-accent" style={{ width: `${active.progress * 100}%` }} /></div></div>}
    {mutation.error && <div className="border-y border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">{mutation.error.message}</div>}
    <EquityChart rows={rows} />
    {latest && <div className="grid gap-4 xl:grid-cols-[0.65fr_1.35fr]">
      <div className="overflow-x-auto border-y border-border bg-surface"><div className="border-b border-border px-3 py-2 text-xs font-medium">月度收益</div><table className="w-full min-w-[260px] text-left text-[11px]"><thead className="bg-elevated/50 text-muted"><tr><th className="px-3 py-2 font-normal">月份</th><th className="px-3 py-2 font-normal">收益</th></tr></thead><tbody>{(metrics.monthly_returns ?? []).map((item: { month: string; return: number }) => <tr key={item.month} className="border-t border-border"><td className="px-3 py-2 font-mono">{item.month}</td><td className={`px-3 py-2 font-mono ${item.return >= 0 ? 'text-bear' : 'text-danger'}`}>{pct(item.return)}</td></tr>)}</tbody></table></div>
      <div className="overflow-x-auto border-y border-border bg-surface"><div className="flex justify-between border-b border-border px-3 py-2"><span className="text-xs font-medium">期末持仓</span><span className="font-mono text-[10px] text-muted">{latestHoldingDate ?? '--'}</span></div><table className="w-full min-w-[620px] text-left text-[11px]"><thead className="bg-elevated/50 text-muted"><tr><th className="px-3 py-2 font-normal">代码</th><th className="px-3 py-2 font-normal">名称</th><th className="px-3 py-2 font-normal">股数</th><th className="px-3 py-2 font-normal">市值</th><th className="px-3 py-2 font-normal">权重</th></tr></thead><tbody>{latestHoldings.map(item => <tr key={item.symbol} className="border-t border-border"><td className="px-3 py-2 font-mono">{item.symbol}</td><td className="px-3 py-2">{item.name || '--'}</td><td className="px-3 py-2 font-mono">{num(item.shares, 0)}</td><td className="px-3 py-2 font-mono">{num(item.market_value, 0)}</td><td className="px-3 py-2 font-mono">{pct(item.weight)}</td></tr>)}</tbody></table>{!latestHoldings.length && <div className="py-8 text-center text-xs text-muted">期末为空仓</div>}</div>
    </div>}
    {latest && <div className="overflow-x-auto border-y border-border bg-surface"><div className="flex justify-between border-b border-border px-3 py-2"><span className="text-xs font-medium">最近成交</span><span className="text-[10px] text-muted">最近 {recentTrades.length} 笔</span></div><table className="w-full min-w-[900px] text-left text-[11px]"><thead className="bg-elevated/50 text-muted"><tr><th className="px-3 py-2 font-normal">成交日</th><th className="px-3 py-2 font-normal">方向</th><th className="px-3 py-2 font-normal">代码</th><th className="px-3 py-2 font-normal">名称</th><th className="px-3 py-2 font-normal">价格</th><th className="px-3 py-2 font-normal">股数</th><th className="px-3 py-2 font-normal">成交额</th><th className="px-3 py-2 font-normal">成本</th><th className="px-3 py-2 font-normal">卖出损益</th></tr></thead><tbody>{recentTrades.map((item, index) => <tr key={`${item.date}-${item.symbol}-${index}`} className="border-t border-border"><td className="px-3 py-2 font-mono text-muted">{item.date}</td><td className={`px-3 py-2 ${item.side === 'buy' ? 'text-danger' : 'text-bear'}`}>{item.side === 'buy' ? '买入' : '卖出'}</td><td className="px-3 py-2 font-mono">{item.symbol}</td><td className="px-3 py-2">{item.name || '--'}</td><td className="px-3 py-2 font-mono">{num(item.price, 3)}</td><td className="px-3 py-2 font-mono">{num(item.shares, 0)}</td><td className="px-3 py-2 font-mono">{num(item.gross_value, 0)}</td><td className="px-3 py-2 font-mono">{num(item.cost, 2)}</td><td className={`px-3 py-2 font-mono ${Number(item.pnl) >= 0 ? 'text-bear' : 'text-danger'}`}>{item.pnl == null ? '--' : num(item.pnl, 2)}</td></tr>)}</tbody></table>{!recentTrades.length && <div className="py-8 text-center text-xs text-muted">暂无成交</div>}</div>}
  </div>
}

function PredictionsView({ detail, onPortfolio }: { detail: QuantModelDetail; onPortfolio: (version: string) => void }) {
  const queryClient = useQueryClient()
  const [selectedDate, setSelectedDate] = useState(detail.prediction_dates[0]?.date ?? '')
  const [search, setSearch] = useState('')
  const [mode, setMode] = useState<'top' | 'bottom'>('top')
  const generate = useMutation({
    mutationFn: () => api.quantGeneratePredictions(detail.version),
    onSuccess: result => {
      setSelectedDate(String(result.date ?? ''))
      void queryClient.invalidateQueries({ queryKey: ['quant', 'model', detail.version] })
      void queryClient.invalidateQueries({ queryKey: ['quant', 'predictions', detail.version] })
    },
  })
  const predictions = useQuery({
    queryKey: ['quant', 'predictions', detail.version, selectedDate, search],
    queryFn: () => api.quantModelPredictions(detail.version, selectedDate || undefined, search, 10_000),
  })
  const allRows = predictions.data?.predictions ?? []
  const rows = (mode === 'top' ? allRows : [...allRows].reverse()).slice(0, 100)
  const summary = predictions.data?.summary
  const exportCsv = async () => {
    const result = await api.quantModelPredictions(detail.version, selectedDate || undefined, search, 10_000)
    const header = 'symbol,name,date,prediction,rank,feature_coverage\n'
    const body = result.predictions.map(item => `${item.symbol},${item.name ?? ''},${item.date},${item.prediction},${item.rank},${item.feature_coverage}`).join('\n')
    const url = URL.createObjectURL(new Blob([header + body], { type: 'text/csv;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${detail.model_id}-${result.date ?? 'prediction'}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }
  return <div className="space-y-4">
    <div className="flex flex-wrap items-end gap-3 border-y border-border bg-surface p-3">
      <label className="min-w-40 text-[11px] text-muted">预测日期<select className={`${inputClass} mt-1`} value={selectedDate} onChange={event => setSelectedDate(event.target.value)}><option value="">最新可用</option>{detail.prediction_dates.map(item => <option key={item.date} value={item.date}>{item.date} · {item.rows} 只</option>)}</select></label>
      <label className="min-w-52 flex-1 text-[11px] text-muted">股票代码或名称<div className="relative mt-1"><Search className="absolute left-2 top-2 h-3.5 w-3.5" /><input className={`${inputClass} pl-7`} value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索代码或名称" /></div></label>
      <button className={actionClass} disabled={generate.isPending || detail.status !== 'published'} onClick={() => generate.mutate()}>{generate.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}生成最新预测</button>
      <button className={quietButton} disabled={!allRows.length} onClick={() => void exportCsv()}><Download className="h-3 w-3" />CSV</button>
      <button className={quietButton} disabled={!allRows.length} onClick={() => onPortfolio(detail.version)}><SlidersHorizontal className="h-3 w-3" />进入组合优化</button>
    </div>
    {generate.error && <div className="border-y border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">{generate.error.message}</div>}
    <div className="grid grid-cols-2 gap-px overflow-hidden border-y border-border bg-border sm:grid-cols-3 xl:grid-cols-6">{[
      ['预测日期', predictions.data?.date ?? '--'], ['有效标的', num(summary?.rows, 0)],
      ['覆盖率', pct(summary?.coverage)], ['预测均值', num(summary?.prediction_mean, 6)],
      ['预测区间', `${num(summary?.prediction_min, 5)} ~ ${num(summary?.prediction_max, 5)}`],
      ['PSI 漂移', num(summary?.psi, 3)],
    ].map(([label, value]) => <div key={label} className="bg-surface p-3"><div className="text-[10px] text-muted">{label}</div><div className="mt-1 font-mono text-sm">{value}</div></div>)}</div>
    {summary?.warnings?.map(item => <div key={item} className="border-y border-warning/30 bg-warning/5 px-3 py-2 text-[11px] text-warning">{item}</div>)}
    <PredictionDistribution rows={allRows} />
    <div className="overflow-x-auto border-y border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-2"><div className="text-xs font-medium">截面预测排名</div><div className="flex rounded-btn border border-border p-0.5">{(['top', 'bottom'] as const).map(item => <button key={item} className={`h-6 px-2 text-[10px] ${mode === item ? 'bg-accent text-white' : 'text-muted'}`} onClick={() => setMode(item)}>{item === 'top' ? 'Top 100' : 'Bottom 100'}</button>)}</div></div>
      <table className="w-full min-w-[780px] text-left text-[11px]"><thead className="bg-elevated/50 text-muted"><tr><th className="px-3 py-2 font-normal">代码</th><th className="px-3 py-2 font-normal">名称</th><th className="px-3 py-2 font-normal">日期</th><th className="px-3 py-2 font-normal">预测值</th><th className="px-3 py-2 font-normal">Rank</th><th className="px-3 py-2 font-normal">特征覆盖</th></tr></thead><tbody>{rows.map(item => <tr key={`${item.date}-${item.symbol}`} className="border-t border-border"><td className="px-3 py-2 font-mono">{item.symbol}</td><td className="px-3 py-2">{item.name || '--'}</td><td className="px-3 py-2 font-mono text-muted">{item.date}</td><td className="px-3 py-2 font-mono">{num(item.prediction, 6)}</td><td className="px-3 py-2 font-mono">{pct(item.rank)}</td><td className="px-3 py-2 font-mono">{pct(item.feature_coverage)}</td></tr>)}</tbody></table>
      {!rows.length && <div className="py-12 text-center text-xs text-muted">尚无盘后预测</div>}
    </div>
  </div>
}

export function ModelCenter({ models, onPortfolio }: { models: QuantModel[]; onPortfolio: (version: string) => void }) {
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState(models[0]?.version ?? '')
  const [view, setView] = useState<DetailView>('validation')
  const [compare, setCompare] = useState<string[]>([])
  const current = selected || models[0]?.version || ''
  const detail = useQuery({
    queryKey: ['quant', 'model', current], queryFn: () => api.quantModelDetail(current),
    enabled: Boolean(current),
    refetchInterval: query => query.state.data?.backtests.some(item => ['queued', 'running', 'cancelling'].includes(item.status)) ? 1500 : false,
  })
  const lifecycle = useMutation({
    mutationFn: async ({ action, model }: { action: 'publish' | 'archive'; model: QuantModelDetail }) => {
      if (action === 'publish' && model.diagnostic.publish_warning && !window.confirm(`当前模型结论为“${gradeLabel[model.diagnostic.grade]}”，仍要发布吗？`)) return null
      return action === 'publish' ? api.quantPublishModel(model.version) : api.quantArchiveModel(model.version)
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['quant', 'models'] })
      void queryClient.invalidateQueries({ queryKey: ['quant', 'model', current] })
    },
  })
  const toggleCompare = (version: string) => setCompare(items => items.includes(version) ? items.filter(item => item !== version) : items.length < 4 ? [...items, version] : items)
  const compared = useMemo(() => models.filter(item => compare.includes(item.version)), [models, compare])
  const model = detail.data
  return <div className="grid gap-4 xl:grid-cols-[330px_1fr]">
    <aside className="min-w-0 border-y border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-3 py-2"><div className="text-xs font-medium">模型版本</div><span className="text-[10px] text-muted">对比 {compare.length}/4</span></div>
      <div className="max-h-[calc(100vh-180px)] overflow-y-auto">{models.map(item => <button key={item.version} onClick={() => setSelected(item.version)} className={`block w-full border-b border-border px-3 py-3 text-left hover:bg-elevated ${current === item.version ? 'bg-elevated/70' : ''}`}><div className="flex items-center gap-2"><span className="min-w-0 flex-1 truncate text-xs font-medium">{item.name}</span><Grade diagnostic={item.diagnostic} /></div><div className="mt-1 truncate font-mono text-[9px] text-muted">{item.version}</div><div className="mt-2 flex items-center justify-between text-[10px] text-secondary"><span>IC {num(item.metrics.rank_ic, 4)}</span><label className="flex items-center gap-1" onClick={event => event.stopPropagation()}><input type="checkbox" checked={compare.includes(item.version)} onChange={() => toggleCompare(item.version)} />对比</label></div></button>)}{!models.length && <div className="py-12 text-center text-xs text-muted">暂无模型</div>}</div>
    </aside>
    <section className="min-w-0 space-y-4">
      {compared.length > 1 && <div className="overflow-x-auto border-y border-border bg-surface"><table className="w-full min-w-[720px] text-left text-[11px]"><thead className="bg-elevated/50 text-muted"><tr><th className="px-3 py-2 font-normal">模型</th><th className="px-3 py-2 font-normal">结论</th><th className="px-3 py-2 font-normal">Rank IC</th><th className="px-3 py-2 font-normal">净 Sharpe</th><th className="px-3 py-2 font-normal">最大回撤</th></tr></thead><tbody>{compared.map(item => <tr key={item.version} className="border-t border-border"><td className="px-3 py-2">{item.name}</td><td className="px-3 py-2"><Grade diagnostic={item.diagnostic} /></td><td className="px-3 py-2 font-mono">{num(item.metrics.rank_ic, 4)}</td><td className="px-3 py-2 font-mono">{num(item.latest_backtest?.metrics.sharpe)}</td><td className="px-3 py-2 font-mono">{pct(item.latest_backtest?.metrics.max_drawdown)}</td></tr>)}</tbody></table></div>}
      {detail.isLoading && <div className="flex justify-center py-24"><Loader2 className="h-5 w-5 animate-spin text-muted" /></div>}
      {detail.error && <div className="border-y border-danger/30 bg-danger/5 px-3 py-3 text-xs text-danger">{detail.error.message}</div>}
      {model && <>
        <div className="flex flex-wrap items-center justify-between gap-3 border-y border-border bg-surface px-3 py-3"><div><div className="flex items-center gap-2"><h2 className="text-sm font-semibold">{model.name}</h2><Grade diagnostic={model.diagnostic} /><span className="text-[10px] text-muted">{model.status}</span></div><div className="mt-1 font-mono text-[10px] text-muted">{model.version}</div></div><div className="flex gap-1">{model.status === 'validated' && <button className={quietButton} disabled={lifecycle.isPending} onClick={() => lifecycle.mutate({ action: 'publish', model })}><Rocket className="h-3 w-3" />发布</button>}{model.status !== 'archived' && <button className={quietButton} disabled={lifecycle.isPending} onClick={() => lifecycle.mutate({ action: 'archive', model })}><Archive className="h-3 w-3" />归档</button>}</div></div>
        <div className="flex max-w-full overflow-x-auto border-b border-border">{DETAIL_TABS.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => setView(id)} className={`inline-flex h-9 shrink-0 items-center gap-1.5 border-b-2 px-3 text-[11px] ${view === id ? 'border-accent text-foreground' : 'border-transparent text-muted'}`}><Icon className="h-3.5 w-3.5" />{label}</button>)}</div>
        {view === 'validation' && <ValidationView detail={model} />}
        {view === 'backtest' && <BacktestView detail={model} />}
        {view === 'predictions' && <PredictionsView detail={model} onPortfolio={onPortfolio} />}
      </>}
    </section>
  </div>
}
