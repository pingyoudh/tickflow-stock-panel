import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  BarChart3, BrainCircuit, Check, Cpu, FlaskConical, Gauge, Info,
  History, Layers3, Library, Loader2, Play, RefreshCw, SlidersHorizontal, Sparkles, Trash2,
} from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { FactorBacktest } from './backtest/FactorBacktest'
import { ModelCenter } from './quant/ModelCenter'
import {
  api, type MLCapabilities, type MLModelSpec, type MLSearchEstimate, type MLSearchSpec,
  type QuantExperiment,
  type QuantFactor, type QuantModel, type QuantStrategy,
} from '@/lib/api'

type Tab = 'factors' | 'research' | 'training' | 'models' | 'strategy' | 'portfolio' | 'experiments'
type Objective = 'equal' | 'score_weight' | 'min_variance' | 'max_sharpe' | 'min_tracking_error'

const TABS: { id: Tab; label: string; icon: typeof Library }[] = [
  { id: 'factors', label: '因子库', icon: Library },
  { id: 'research', label: '因子研究', icon: FlaskConical },
  { id: 'training', label: '模型训练', icon: BrainCircuit },
  { id: 'models', label: '模型中心', icon: Layers3 },
  { id: 'strategy', label: '策略模型', icon: BarChart3 },
  { id: 'portfolio', label: '组合优化', icon: SlidersHorizontal },
  { id: 'experiments', label: '实验记录', icon: History },
]

const inputClass = 'h-8 w-full rounded-input border border-border bg-base px-2 text-xs text-foreground outline-none focus:border-accent'
const actionClass = 'inline-flex h-8 items-center justify-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-50'
const quietButton = 'inline-flex h-7 items-center justify-center gap-1 rounded-btn border border-border bg-base px-2 text-[11px] text-secondary hover:bg-elevated hover:text-foreground disabled:opacity-40'

function isoDate(offsetYears = 0) {
  const value = new Date()
  value.setFullYear(value.getFullYear() + offsetYears)
  return value.toISOString().slice(0, 10)
}

function metric(value: unknown, digits = 4) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(digits) : '--'
}

function pct(value: unknown, digits = 2) {
  const number = Number(value)
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : '--'
}

function Status({ value }: { value: string }) {
  const color = value === 'published' || value === 'completed'
    ? 'text-bear border-bear/30 bg-bear/10'
    : value === 'failed' || value === 'archived'
      ? 'text-danger border-danger/30 bg-danger/10'
      : 'text-warning border-warning/30 bg-warning/10'
  return <span className={`rounded border px-1.5 py-0.5 text-[10px] ${color}`}>{value}</span>
}

function FactorLibrary({ factors }: { factors: QuantFactor[] }) {
  const groups = useMemo(() => ({
    builtin: factors.filter(item => item.authoring_type === 'builtin'),
    custom: factors.filter(item => item.authoring_type === 'declarative' || item.authoring_type === 'python'),
    model: factors.filter(item => item.authoring_type === 'model'),
  }), [factors])
  return (
    <div className="overflow-hidden border-y border-border bg-surface">
      <div className="grid grid-cols-[minmax(180px,1.2fr)_100px_110px_minmax(220px,2fr)_120px] border-b border-border bg-elevated/50 px-3 py-2 text-[11px] text-muted">
        <span>因子</span><span>类型</span><span>资产</span><span>版本 / 描述</span><span>状态</span>
      </div>
      {(['builtin', 'custom', 'model'] as const).map(group => groups[group].map(item => (
        <div key={item.id} className="grid grid-cols-[minmax(180px,1.2fr)_100px_110px_minmax(220px,2fr)_120px] items-center border-b border-border/70 px-3 py-2.5 text-xs last:border-0">
          <div className="min-w-0"><div className="truncate font-medium">{item.name}</div><div className="truncate font-mono text-[10px] text-muted">{item.id}</div></div>
          <span className="text-secondary">{item.authoring_type}</span>
          <span className="text-secondary">{item.asset_types.join(' / ')}</span>
          <div className="min-w-0"><div className="truncate font-mono text-[10px] text-secondary">{item.version}</div><div className="truncate text-[11px] text-muted">{item.description}</div></div>
          <span className="text-[11px] text-secondary">{item.authoring_type === 'python' ? (item.trusted ? '已信任' : '未启用') : item.readonly ? '只读' : '可编辑'}</span>
        </div>
      )))}
      {factors.length === 0 && <div className="px-3 py-10 text-center text-xs text-muted">暂无因子</div>}
    </div>
  )
}

function CapabilityBar({ data }: { data?: MLCapabilities }) {
  const gpu = data?.gpu
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-y border-border bg-surface px-3 py-2 text-xs">
      <span className="inline-flex items-center gap-1.5"><Gauge className="h-3.5 w-3.5 text-accent" />{gpu?.available ? `${gpu.name} · ${gpu.memory_mb} MB` : 'GPU 未探测到'}</span>
      <span className="inline-flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5 text-secondary" />CPU 线程 {data?.cpu_threads ?? '--'}</span>
      {Object.entries(data?.algorithms ?? {}).map(([name, item]) => (
        <span key={name} className={item.installed ? 'text-bear' : 'text-warning'}>{name} {item.installed ? item.version : '未安装'}</span>
      ))}
    </div>
  )
}

function SmartTrainingPanel({ factors, busy, onEstimate, onSearch }: {
  factors: QuantFactor[]
  busy: boolean
  onEstimate: (spec: MLSearchSpec) => Promise<MLSearchEstimate>
  onSearch: (spec: MLSearchSpec) => void
}) {
  const candidates = factors.filter(item => item.authoring_type !== 'model' && item.asset_types.includes('stock'))
  const [roles, setRoles] = useState<Record<string, 'auto' | 'required' | 'excluded'>>({})
  const [name, setName] = useState('A股多因子智能模型')
  const [symbols, setSymbols] = useState('')
  const [start, setStart] = useState(isoDate(-5))
  const [end, setEnd] = useState(isoDate())
  const [horizon, setHorizon] = useState<1 | 5 | 10 | 20>(5)
  const [benchmark, setBenchmark] = useState('000300.SH')
  const [budget, setBudget] = useState<'quick' | 'standard' | 'overnight'>('standard')
  const [device, setDevice] = useState<'auto' | 'cpu' | 'gpu'>('auto')
  const [estimate, setEstimate] = useState<MLSearchEstimate | null>(null)
  const [estimateError, setEstimateError] = useState('')
  const references = candidates.map(item => ({ id: item.id, version: item.version }))
  const required = candidates.filter(item => roles[item.id] === 'required').map(item => ({ id: item.id, version: item.version }))
  const excluded = candidates.filter(item => roles[item.id] === 'excluded').map(item => ({ id: item.id, version: item.version }))
  const availableCount = candidates.length - excluded.length
  const minFeatures = Math.min(8, Math.max(1, availableCount))
  const maxFeatures = Math.min(30, Math.max(minFeatures, availableCount))
  const buildSpec = (): MLSearchSpec => ({
    id: `automl_${Date.now()}`, name, asset_type: 'stock',
    symbols: symbols.trim() ? symbols.split(/[,，\s]+/).filter(Boolean) : null,
    start, end, target: { horizon, benchmark_mode: 'index', benchmark_symbol: benchmark },
    factor_pool: references, required_factors: required, excluded_factors: excluded,
    algorithms: ['elastic_net', 'lightgbm', 'xgboost'], budget,
    min_features: minFeatures, max_features: maxFeatures, shortlist_limit: 80,
    inner_folds: 3, inner_validation_days: 63,
    walk_forward: { train_days: 756, validation_days: 126, test_days: 126, step_days: 126 },
    costs: { top_n: 10, commission_pct: 0.0002, stamp_tax_pct: 0.0005, slippage_bps: 5 },
    device, seed: 42, universe_filters: {},
  })
  const cycleRole = (id: string) => setRoles(current => ({
    ...current,
    [id]: current[id] === 'required' ? 'excluded' : current[id] === 'excluded' ? 'auto' : 'required',
  }))
  const estimateResources = async () => {
    setEstimateError('')
    try { setEstimate(await onEstimate(buildSpec())) } catch (error) { setEstimateError(error instanceof Error ? error.message : String(error)) }
  }
  return <div className="grid gap-4 border-y border-border bg-surface p-3 xl:grid-cols-[380px_1fr]">
    <div className="grid grid-cols-2 gap-3 content-start">
      <label className="col-span-2 text-[11px] text-muted">模型名称<input className={`${inputClass} mt-1`} value={name} onChange={event => setName(event.target.value)} /></label>
      <label className="text-[11px] text-muted">开始日期<input type="date" className={`${inputClass} mt-1`} value={start} onChange={event => setStart(event.target.value)} /></label>
      <label className="text-[11px] text-muted">结束日期<input type="date" className={`${inputClass} mt-1`} value={end} onChange={event => setEnd(event.target.value)} /></label>
      <label className="text-[11px] text-muted">预测周期<select className={`${inputClass} mt-1`} value={horizon} onChange={event => setHorizon(Number(event.target.value) as typeof horizon)}>{[1, 5, 10, 20].map(value => <option key={value} value={value}>{value} 个交易日</option>)}</select></label>
      <label className="text-[11px] text-muted">计算预算<select className={`${inputClass} mt-1`} value={budget} onChange={event => setBudget(event.target.value as typeof budget)}><option value="quick">快速 · 8 试验</option><option value="standard">标准 · 72 试验</option><option value="overnight">隔夜 · 180 试验</option></select></label>
      <label className="text-[11px] text-muted">基准<input className={`${inputClass} mt-1 font-mono`} value={benchmark} onChange={event => setBenchmark(event.target.value)} /></label>
      <label className="text-[11px] text-muted">设备<select className={`${inputClass} mt-1`} value={device} onChange={event => setDevice(event.target.value as typeof device)}><option value="auto">自动回退</option><option value="gpu">GPU 优先</option><option value="cpu">CPU</option></select></label>
      <label className="col-span-2 text-[11px] text-muted">股票池（留空为全市场）<input className={`${inputClass} mt-1 font-mono`} value={symbols} onChange={event => setSymbols(event.target.value)} placeholder="600000.SH, 000001.SZ" /></label>
      <div className="col-span-2 grid grid-cols-2 gap-2"><button className={quietButton} disabled={busy || references.length === 0} onClick={() => void estimateResources()}><Info className="h-3 w-3" />估算资源</button><button className={actionClass} disabled={busy || availableCount < minFeatures || !name.trim()} onClick={() => onSearch(buildSpec())}>{busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}开始智能训练</button></div>
      {estimate && <div className="col-span-2 grid grid-cols-2 gap-px border border-border bg-border text-[10px]"><div className="bg-base p-2"><span className="text-muted">预计面板</span><div className="mt-1 font-mono">{estimate.estimated_rows.toLocaleString()} 行</div></div><div className="bg-base p-2"><span className="text-muted">锁定因子</span><div className="mt-1 font-mono">{estimate.factor_count} 个</div></div><div className="bg-base p-2"><span className="text-muted">每窗口候选</span><div className="mt-1 font-mono">{estimate.search_trials_per_window} 次</div></div><div className="bg-base p-2"><span className="text-muted">模型拟合</span><div className="mt-1 font-mono">{estimate.estimated_model_fits} 次</div></div><div className="bg-base p-2"><span className="text-muted">外层测试折</span><div className="mt-1 font-mono">{estimate.outer_folds}</div></div><div className="bg-base p-2"><span className="text-muted">预计耗时</span><div className="mt-1 font-mono">约 {estimate.estimated_hours} 小时</div></div></div>}
      {estimateError && <div className="col-span-2 text-[11px] text-danger">{estimateError}</div>}
      {estimate?.warnings.map(item => <div key={item} className="col-span-2 text-[10px] text-warning">{item}</div>)}
    </div>
    <div className="min-w-0">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted"><span>因子池 · {candidates.length} 个</span><div className="flex gap-3"><span>自动 {candidates.filter(item => !roles[item.id] || roles[item.id] === 'auto').length}</span><span className="text-bear">必选 {required.length}</span><span className="text-danger">排除 {excluded.length}</span></div></div>
      <div className="grid grid-cols-2 gap-px overflow-hidden border border-border bg-border lg:grid-cols-3">
        {candidates.map(item => { const role = roles[item.id] ?? 'auto'; return <button key={`${item.id}-${item.version}`} title={item.point_in_time === false ? '该因子不是时点正确数据，智能训练会将其淘汰' : '点击切换：自动 → 必选 → 排除'} onClick={() => cycleRole(item.id)} className={`flex h-12 min-w-0 items-center justify-between bg-surface px-2 text-left text-[11px] hover:bg-elevated ${role === 'required' ? 'text-bear' : role === 'excluded' || item.point_in_time === false ? 'text-danger' : 'text-secondary'}`}><span className="min-w-0"><span className="block truncate">{item.name}</span><span className="block truncate font-mono text-[9px] text-muted">{item.family || 'other'} · {item.version}</span></span><span className="ml-2 shrink-0 text-[9px]">{item.point_in_time === false ? '非时点' : role === 'required' ? '必选' : role === 'excluded' ? '排除' : '自动'}</span></button> })}
      </div>
    </div>
  </div>
}

function ManualTrainingPanel({ factors, busy, onTrain }: {
  factors: QuantFactor[]
  busy: boolean
  onTrain: (spec: MLModelSpec) => void
}) {
  const candidates = factors.filter(item => item.authoring_type !== 'model' && item.asset_types.includes('stock'))
  const [algorithm, setAlgorithm] = useState<'lightgbm' | 'xgboost'>('xgboost')
  const [device, setDevice] = useState<'auto' | 'cpu' | 'gpu'>('auto')
  const [features, setFeatures] = useState<string[]>(['momentum_20d', 'annual_vol_20d', 'rsi_14', 'turnover_rate'])
  const [name, setName] = useState('A股多因子收益模型')
  const [symbols, setSymbols] = useState('')
  const [start, setStart] = useState(isoDate(-5))
  const [end, setEnd] = useState(isoDate())
  const [horizon, setHorizon] = useState<1 | 5 | 10 | 20>(5)
  const [benchmark, setBenchmark] = useState('000300.SH')
  const [tuning, setTuning] = useState(false)

  const submit = () => onTrain({
    id: `model_${Date.now()}`,
    name, algorithm, asset_type: 'stock',
    symbols: symbols.trim() ? symbols.split(/[,，\s]+/).filter(Boolean) : null,
    features, start, end,
    target: { horizon, benchmark_mode: 'index', benchmark_symbol: benchmark },
    walk_forward: { train_days: 756, validation_days: 126, test_days: 126, step_days: 126 },
    tuning: { enabled: tuning, max_trials: 20 },
    device, params: {}, seed: 42, universe_filters: {},
  })

  return (
    <div className="grid gap-4 border-y border-border bg-surface p-3 xl:grid-cols-[360px_1fr]">
        <div className="grid grid-cols-2 gap-3 content-start">
          <label className="col-span-2 text-[11px] text-muted">模型名称<input className={`${inputClass} mt-1`} value={name} onChange={e => setName(e.target.value)} /></label>
          <label className="text-[11px] text-muted">算法<select className={`${inputClass} mt-1`} value={algorithm} onChange={e => setAlgorithm(e.target.value as typeof algorithm)}><option value="lightgbm">LightGBM</option><option value="xgboost">XGBoost</option></select></label>
          <label className="text-[11px] text-muted">设备<select className={`${inputClass} mt-1`} value={device} onChange={e => setDevice(e.target.value as typeof device)}><option value="auto">自动回退</option><option value="gpu">GPU 优先</option><option value="cpu">CPU</option></select></label>
          <label className="text-[11px] text-muted">开始日期<input type="date" className={`${inputClass} mt-1`} value={start} onChange={e => setStart(e.target.value)} /></label>
          <label className="text-[11px] text-muted">结束日期<input type="date" className={`${inputClass} mt-1`} value={end} onChange={e => setEnd(e.target.value)} /></label>
          <label className="text-[11px] text-muted">预测周期<select className={`${inputClass} mt-1`} value={horizon} onChange={e => setHorizon(Number(e.target.value) as typeof horizon)}>{[1, 5, 10, 20].map(value => <option key={value} value={value}>{value} 个交易日</option>)}</select></label>
          <label className="text-[11px] text-muted">基准<input className={`${inputClass} mt-1 font-mono`} value={benchmark} onChange={e => setBenchmark(e.target.value)} /></label>
          <label className="col-span-2 text-[11px] text-muted">股票池（留空为全市场）<input className={`${inputClass} mt-1 font-mono`} value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="600000.SH, 000001.SZ" /></label>
          <label className="col-span-2 flex items-center gap-2 text-xs text-secondary"><input type="checkbox" checked={tuning} onChange={e => setTuning(e.target.checked)} />Optuna · 每折最多 20 次</label>
          <button className={`${actionClass} col-span-2`} disabled={busy || features.length === 0 || !name.trim()} onClick={submit}>{busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}开始 Walk-forward 训练</button>
        </div>
        <div>
          <div className="mb-2 flex items-center justify-between text-[11px] text-muted"><span>输入因子</span><span>{features.length} 个</span></div>
          <div className="grid grid-cols-2 gap-px overflow-hidden border border-border bg-border lg:grid-cols-3">
            {candidates.map(item => {
              const checked = features.includes(item.id)
              return <button key={item.id} onClick={() => setFeatures(current => checked ? current.filter(id => id !== item.id) : [...current, item.id])} className={`flex h-10 items-center justify-between bg-surface px-2 text-left text-[11px] hover:bg-elevated ${checked ? 'text-foreground' : 'text-muted'}`}><span className="truncate">{item.name}</span>{checked && <Check className="h-3.5 w-3.5 shrink-0 text-accent" />}</button>
            })}
          </div>
        </div>
    </div>
  )
}

function TrainingWorkspace({ factors, capabilities, trainBusy, searchBusy, onTrain, onEstimate, onSearch }: {
  factors: QuantFactor[]
  capabilities?: MLCapabilities
  trainBusy: boolean
  searchBusy: boolean
  onTrain: (spec: MLModelSpec) => void
  onEstimate: (spec: MLSearchSpec) => Promise<MLSearchEstimate>
  onSearch: (spec: MLSearchSpec) => void
}) {
  const [mode, setMode] = useState<'smart' | 'manual'>('smart')
  return <div className="space-y-3"><CapabilityBar data={capabilities} /><div className="flex border-b border-border"><button className={`inline-flex h-8 items-center gap-1.5 border-b-2 px-3 text-[11px] ${mode === 'smart' ? 'border-accent text-foreground' : 'border-transparent text-muted'}`} onClick={() => setMode('smart')}><Sparkles className="h-3.5 w-3.5" />智能训练</button><button className={`inline-flex h-8 items-center gap-1.5 border-b-2 px-3 text-[11px] ${mode === 'manual' ? 'border-accent text-foreground' : 'border-transparent text-muted'}`} onClick={() => setMode('manual')}><SlidersHorizontal className="h-3.5 w-3.5" />手工训练</button></div>{mode === 'smart' ? <SmartTrainingPanel factors={factors} busy={searchBusy} onEstimate={onEstimate} onSearch={onSearch} /> : <ManualTrainingPanel factors={factors} busy={trainBusy} onTrain={onTrain} />}</div>
}

function StrategyPanel({ factors, strategies, busy, onSave, onDelete }: {
  factors: QuantFactor[]
  strategies: QuantStrategy[]
  busy: boolean
  onSave: (spec: QuantStrategy) => void
  onDelete: (id: string) => void
}) {
  const available = factors.filter(item => item.asset_types.includes('stock'))
  const [name, setName] = useState('多因子选股策略')
  const [selected, setSelected] = useState<string[]>([])
  const [topN, setTopN] = useState(10)
  const [rebalance, setRebalance] = useState<'daily' | 'weekly' | 'monthly'>('weekly')
  const save = () => {
    const references = selected.map(id => available.find(item => item.id === id)).filter((item): item is QuantFactor => Boolean(item))
    onSave({
      id: `strategy_${Date.now()}`, name, asset_type: 'stock', symbols: null,
      factors: references.map(item => ({ factor_id: item.id, factor_version: item.version, weight: 1 / references.length })),
      candidate_mode: 'top_n', score_threshold: null, top_n: topN, rebalance,
      entry_rule: 'next_open', exit_rule: 'rebalance',
    })
  }
  return <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
    <div className="border-y border-border bg-surface p-3"><div className="grid grid-cols-2 gap-3"><label className="col-span-2 text-[11px] text-muted">策略名称<input className={`${inputClass} mt-1`} value={name} onChange={event => setName(event.target.value)} /></label><label className="text-[11px] text-muted">候选数量<input type="number" min={1} max={500} className={`${inputClass} mt-1`} value={topN} onChange={event => setTopN(Number(event.target.value))} /></label><label className="text-[11px] text-muted">调仓<select className={`${inputClass} mt-1`} value={rebalance} onChange={event => setRebalance(event.target.value as typeof rebalance)}><option value="daily">每日</option><option value="weekly">每周</option><option value="monthly">每月</option></select></label></div><div className="mb-1 mt-3 flex justify-between text-[11px] text-muted"><span>版本锁定因子</span><span>{selected.length}</span></div><div className="max-h-64 overflow-y-auto border border-border">{available.map(item => { const checked = selected.includes(item.id); return <button key={`${item.id}-${item.version}`} onClick={() => setSelected(current => checked ? current.filter(id => id !== item.id) : [...current, item.id])} className="flex w-full items-center justify-between border-b border-border px-2 py-2 text-left text-[11px] last:border-0 hover:bg-elevated"><span className="min-w-0"><span className="block truncate">{item.name}</span><span className="block truncate font-mono text-[9px] text-muted">{item.version}</span></span>{checked && <Check className="h-3.5 w-3.5 shrink-0 text-accent" />}</button> })}</div><button className={`${actionClass} mt-3 w-full`} disabled={busy || selected.length === 0 || !name.trim()} onClick={save}>保存策略版本</button></div>
    <div className="space-y-2">{strategies.map(item => <div key={item.id} className="border-y border-border bg-surface px-3 py-2.5"><div className="flex items-center justify-between"><div><div className="text-xs font-medium">{item.name}</div><div className="mt-0.5 font-mono text-[10px] text-muted">{item.id}</div></div><button className={quietButton} disabled={busy} onClick={() => onDelete(item.id)} title="删除"><Trash2 className="h-3 w-3" /></button></div><div className="mt-2 flex flex-wrap gap-1">{item.factors.map(factor => <span key={factor.factor_id} className="rounded border border-border px-1.5 py-0.5 font-mono text-[9px] text-secondary">{factor.factor_id} · {(factor.weight * 100).toFixed(0)}%</span>)}</div><div className="mt-2 text-[10px] text-muted">Top {item.top_n} · {item.rebalance} · next_open</div></div>)}{strategies.length === 0 && <div className="border-y border-border bg-surface py-12 text-center text-xs text-muted">暂无策略版本</div>}</div>
  </div>
}

function PortfolioPanel({ models, initialVersion = '' }: { models: QuantModel[]; initialVersion?: string }) {
  const published = models.filter(item => item.status === 'published')
  const [version, setVersion] = useState(initialVersion)
  const [objective, setObjective] = useState<Objective>('score_weight')
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [warnings, setWarnings] = useState<string[]>([])
  const selected = version || published[0]?.version || ''
  const optimize = useMutation({
    mutationFn: () => api.quantOptimizePortfolio({ model_version: selected, objective, max_positions: 10, max_weight: 0.2, industry_cap: 0.3, turnover_cap: 0.5 }),
    onSuccess: result => { setWeights(result.weights); setWarnings(result.warnings) },
  })
  return <div className="grid gap-4 border-y border-border bg-surface p-3 lg:grid-cols-[340px_1fr]">
    <div className="space-y-3"><label className="block text-[11px] text-muted">发布模型<select className={`${inputClass} mt-1`} value={selected} onChange={e => setVersion(e.target.value)}>{published.map(item => <option key={item.version} value={item.version}>{item.name} · {item.version.slice(-8)}</option>)}</select></label><div><div className="mb-1 text-[11px] text-muted">优化目标</div><div className="grid grid-cols-2 gap-1">{(['equal', 'score_weight', 'min_variance', 'max_sharpe', 'min_tracking_error'] as Objective[]).map(item => <button key={item} onClick={() => setObjective(item)} className={`${quietButton} ${objective === item ? 'border-accent text-accent' : ''}`}>{item}</button>)}</div></div><button className={`${actionClass} w-full`} disabled={!selected || optimize.isPending} onClick={() => optimize.mutate()}>{optimize.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <SlidersHorizontal className="h-3.5 w-3.5" />}计算权重</button>{warnings.map(item => <div key={item} className="text-[11px] text-warning">{item}</div>)}</div>
    <div className="min-w-0"><div className="mb-2 text-[11px] text-muted">目标权重</div>{Object.entries(weights).sort((a, b) => b[1] - a[1]).map(([symbol, weight]) => <div key={symbol} className="grid grid-cols-[100px_1fr_60px] items-center gap-2 border-b border-border py-2 text-xs"><span className="font-mono">{symbol}</span><div className="h-1.5 bg-elevated"><div className="h-full bg-accent" style={{ width: `${Math.min(100, weight * 500)}%` }} /></div><span className="text-right font-mono">{(weight * 100).toFixed(2)}%</span></div>)}{Object.keys(weights).length === 0 && <div className="py-12 text-center text-xs text-muted">暂无优化结果</div>}</div>
  </div>
}

function SearchResult({ result }: { result: Record<string, any> }) {
  const funnel = result.factor_funnel ?? {}
  const champion = result.champion ?? {}
  const leaderboard = (result.candidate_leaderboard ?? []) as Record<string, any>[]
  const quality = (result.factor_quality ?? []) as Record<string, any>[]
  const frequencies = Object.entries(result.feature_selection_frequency ?? {})
    .sort((left, right) => Number(right[1]) - Number(left[1])).slice(0, 12)
  const reason = (item: Record<string, any>) => item.reason || (item.status === 'accepted' ? '通过质量筛选' : item.error || '--')
  return <div className="mt-3 border-t border-border pt-3">
    <div className="grid grid-cols-2 gap-px border border-border bg-border md:grid-cols-4">
      {[['入池', funnel.submitted], ['质量通过', funnel.quality_passed], ['短名单', funnel.shortlisted], ['冠军采用', funnel.selected]].map(([label, value]) => <div key={String(label)} className="bg-base p-2"><div className="text-[10px] text-muted">{label}</div><div className="mt-1 font-mono text-sm">{value ?? '--'}</div></div>)}
    </div>
    <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px]"><span>冠军算法 <b className="font-mono">{champion.algorithm ?? '--'}</b></span><span>因子数 <b className="font-mono">{champion.features?.length ?? '--'}</b></span><span title="仅用于内层候选选择，不是严格 OOS 绩效">内层选择分 <b className="font-mono">{metric(champion.selection_score)}</b></span><span title="外层测试折拼接出的严格样本外结果">严格 OOS Rank IC <b className="font-mono">{metric(result.metrics?.rank_ic)}</b></span></div>
    {(result.warnings ?? []).map((item: string) => <div key={item} className="mt-2 text-[10px] text-warning">{item}</div>)}
    <div className="mt-4 grid gap-4 xl:grid-cols-2">
      <div className="min-w-0"><div className="mb-1 text-[11px] font-medium">候选榜 <span className="font-normal text-muted">内层验证指标</span></div><div className="overflow-x-auto border border-border"><table className="w-full min-w-[620px] text-[10px]"><thead className="bg-elevated text-muted"><tr><th className="px-2 py-1.5 text-left">算法</th><th>因子</th><th title="综合内层验证指标并扣除复杂度惩罚">选择分</th><th>Rank IC</th><th>ICIR</th><th>较差净超额</th><th>Sharpe</th></tr></thead><tbody>{leaderboard.slice(0, 10).map((item, index) => <tr key={`${item.algorithm}-${item.trial}-${index}`} className="border-t border-border"><td className="px-2 py-1.5 font-mono">{item.algorithm}</td><td className="text-center">{item.features?.length ?? 0}</td><td className="text-center font-mono">{metric(item.score)}</td><td className="text-center font-mono">{metric(item.metrics?.rank_ic)}</td><td className="text-center font-mono">{metric(item.metrics?.icir)}</td><td className="text-center font-mono">{pct(Math.min(Number(item.economic?.annual_excess_vs_index ?? 0), Number(item.economic?.annual_excess_vs_universe ?? 0)))}</td><td className="text-center font-mono">{metric(item.economic?.sharpe, 2)}</td></tr>)}</tbody></table></div></div>
      <div className="min-w-0"><div className="mb-1 text-[11px] font-medium">外层选择频率 <span className="font-normal text-muted">每折冠军采用次数</span></div><div className="border border-border px-2">{frequencies.map(([name, value]) => <div key={name} className="grid grid-cols-[150px_1fr_42px] items-center gap-2 border-b border-border py-1.5 text-[10px] last:border-0"><span className="truncate font-mono">{name}</span><div className="h-1.5 bg-elevated"><div className="h-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }} /></div><span className="text-right font-mono">{pct(value, 0)}</span></div>)}</div></div>
    </div>
    <div className="mt-4"><div className="mb-1 text-[11px] font-medium">因子漏斗明细 <span className="font-normal text-muted">不以单因子 IC 作为硬门槛</span></div><div className="max-h-72 overflow-auto border border-border"><table className="w-full min-w-[680px] text-[10px]"><thead className="sticky top-0 bg-elevated text-muted"><tr><th className="px-2 py-1.5 text-left">因子</th><th>状态</th><th>覆盖率</th><th>Rank IC</th><th>ICIR</th><th className="px-2 text-left">原因</th></tr></thead><tbody>{quality.map(item => <tr key={item.factor_id} className="border-t border-border"><td className="px-2 py-1.5 font-mono">{item.factor_id}</td><td className={`text-center ${item.status === 'accepted' ? 'text-bear' : 'text-danger'}`}>{item.status === 'accepted' ? '保留' : '淘汰'}</td><td className="text-center font-mono">{pct(item.coverage)}</td><td className="text-center font-mono">{metric(item.rank_ic)}</td><td className="text-center font-mono">{metric(item.icir)}</td><td className="px-2 text-muted">{reason(item)}</td></tr>)}</tbody></table></div></div>
  </div>
}

function ExperimentsPanel({ experiments, busy, onAction }: { experiments: QuantExperiment[]; busy: boolean; onAction: (action: 'cancel' | 'rerun' | 'delete', id: string) => void }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  return <div className="space-y-2">{experiments.map(item => { const active = ['queued', 'running', 'cancelling'].includes(item.status); const metrics = item.result?.metrics ?? {}; const search = item.kind === 'ml_search'; return <div key={item.run_id} className="border-y border-border bg-surface px-3 py-2.5"><div className="flex flex-wrap items-center justify-between gap-2"><div><div className="flex items-center gap-2 text-xs font-medium">{item.spec?.name ?? item.kind}<Status value={item.status} />{item.input_changed && <span className="rounded border border-warning/30 bg-warning/10 px-1.5 py-0.5 text-[10px] text-warning">输入已变化</span>}</div><div className="mt-1 font-mono text-[10px] text-muted">{item.run_id} · {new Date(item.created_at).toLocaleString()}</div></div><div className="flex gap-1">{item.status === 'completed' && search && <button className={quietButton} onClick={() => setExpanded(current => current === item.run_id ? null : item.run_id)}>{expanded === item.run_id ? '收起结果' : '查看结果'}</button>}{active && <button className={quietButton} disabled={busy || item.status === 'cancelling'} onClick={() => onAction('cancel', item.run_id)}>{item.status === 'cancelling' ? '正在取消' : '取消'}</button>}{!active && <button className={quietButton} disabled={busy} onClick={() => onAction('rerun', item.run_id)}><RefreshCw className="h-3 w-3" />重跑</button>}{!active && <button className={quietButton} disabled={busy} onClick={() => onAction('delete', item.run_id)} title="删除"><Trash2 className="h-3 w-3" /></button>}</div></div><div className="mt-2 h-1 overflow-hidden bg-elevated"><div className={`h-full ${item.status === 'failed' ? 'bg-danger' : 'bg-accent'}`} style={{ width: `${item.progress * 100}%` }} /></div><div className="mt-1 flex justify-between text-[10px] text-muted"><span>{item.error || item.message}</span><span>{Math.round(item.progress * 100)}%</span></div>{item.status === 'completed' && (search ? <div className="mt-2 flex flex-wrap gap-5 border-t border-border pt-2 text-[11px]"><span>冠军 <b className="font-mono">{item.result?.champion?.algorithm ?? '--'}</b></span><span>因子 <b className="font-mono">{item.result?.champion?.features?.length ?? '--'}</b></span><span title="内层验证选择分">选择分 <b className="font-mono">{metric(item.result?.champion?.selection_score)}</b></span><span title="严格外层 OOS 指标">OOS Rank IC <b className="font-mono">{metric(metrics.rank_ic)}</b></span></div> : item.kind === 'ml_backtest' ? <div className="mt-2 flex flex-wrap gap-5 border-t border-border pt-2 text-[11px]"><span>累计收益 <b className="font-mono">{pct(metrics.total_return)}</b></span><span>Sharpe <b className="font-mono">{metric(metrics.sharpe)}</b></span><span>最大回撤 <b className="font-mono">{pct(metrics.max_drawdown)}</b></span><span>指数超额 <b className="font-mono">{pct(metrics.excess_vs_index)}</b></span></div> : <div className="mt-2 flex flex-wrap gap-5 border-t border-border pt-2 text-[11px]"><span>Rank IC <b className="font-mono">{metric(metrics.rank_ic)}</b></span><span>ICIR <b className="font-mono">{metric(metrics.icir)}</b></span><span>OOS RMSE <b className="font-mono">{metric(metrics.rmse)}</b></span></div>)}{expanded === item.run_id && search && <SearchResult result={item.result} />}</div> })}{experiments.length === 0 && <div className="border-y border-border bg-surface py-12 text-center text-xs text-muted">暂无实验</div>}</div>
}

export function Quant() {
  const [tab, setTab] = useState<Tab>('training')
  const [portfolioVersion, setPortfolioVersion] = useState('')
  const queryClient = useQueryClient()
  const factors = useQuery({ queryKey: ['quant', 'factors'], queryFn: api.quantFactors })
  const capabilities = useQuery({ queryKey: ['quant', 'capabilities'], queryFn: api.quantMLCapabilities })
  const models = useQuery({ queryKey: ['quant', 'models'], queryFn: api.quantModels })
  const strategies = useQuery({ queryKey: ['quant', 'strategies'], queryFn: api.quantStrategies })
  const experiments = useQuery({ queryKey: ['quant', 'experiments'], queryFn: api.quantExperiments, refetchInterval: query => query.state.data?.experiments.some(item => ['queued', 'running', 'cancelling'].includes(item.status)) ? 1500 : false })
  const train = useMutation({ mutationFn: api.quantTrain, onSuccess: () => { void queryClient.invalidateQueries({ queryKey: ['quant', 'experiments'] }); setTab('experiments') } })
  const search = useMutation({ mutationFn: api.quantSearch, onSuccess: () => { void queryClient.invalidateQueries({ queryKey: ['quant', 'experiments'] }); setTab('experiments') } })
  const experimentAction = useMutation<unknown, Error, { action: 'cancel' | 'rerun' | 'delete'; id: string }>({ mutationFn: ({ action, id }) => action === 'cancel' ? api.quantCancelExperiment(id) : action === 'rerun' ? api.quantRerunExperiment(id) : api.quantDeleteExperiment(id), onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['quant', 'experiments'] }) })
  const strategyAction = useMutation<unknown, Error, { action: 'save' | 'delete'; spec?: QuantStrategy; id?: string }>({ mutationFn: ({ action, spec, id }) => action === 'save' ? api.quantSaveStrategy(spec!) : api.quantDeleteStrategy(id!), onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['quant', 'strategies'] }) })
  const tabSwitch = <div className="flex max-w-full overflow-x-auto rounded-btn border border-border bg-surface p-0.5">{TABS.map(item => { const Icon = item.icon; return <button key={item.id} onClick={() => setTab(item.id)} className={`inline-flex h-7 shrink-0 items-center gap-1.5 rounded-[5px] px-2.5 text-[11px] ${tab === item.id ? 'bg-accent text-white' : 'text-secondary hover:bg-elevated'}`}><Icon className="h-3.5 w-3.5" />{item.label}</button> })}</div>
  const factorRows = factors.data?.factors ?? []
  const modelRows = models.data?.models ?? []
  const openPortfolio = (version: string) => { setPortfolioVersion(version); setTab('portfolio') }
  return <div className="min-h-full bg-base"><PageHeader title="量化研究" subtitle="因子 · 机器学习 · 组合" right={tabSwitch} /><main className="p-3 lg:p-4">{tab === 'factors' && <FactorLibrary factors={factorRows} />}{tab === 'research' && <FactorBacktest />}{tab === 'training' && <TrainingWorkspace factors={factorRows} capabilities={capabilities.data} trainBusy={train.isPending} searchBusy={search.isPending} onTrain={spec => train.mutate(spec)} onEstimate={api.quantSearchEstimate} onSearch={spec => search.mutate(spec)} />}{tab === 'models' && <ModelCenter models={modelRows} onPortfolio={openPortfolio} />}{tab === 'strategy' && <StrategyPanel factors={factorRows} strategies={strategies.data?.strategies ?? []} busy={strategyAction.isPending} onSave={spec => strategyAction.mutate({ action: 'save', spec })} onDelete={id => strategyAction.mutate({ action: 'delete', id })} />}{tab === 'portfolio' && <PortfolioPanel key={portfolioVersion} models={modelRows} initialVersion={portfolioVersion} />}{tab === 'experiments' && <ExperimentsPanel experiments={experiments.data?.experiments ?? []} busy={experimentAction.isPending} onAction={(action, id) => experimentAction.mutate({ action, id })} />}</main></div>
}
