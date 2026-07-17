import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { Quant } from './Quant'

const mocks = vi.hoisted(() => ({
  train: vi.fn(),
  search: vi.fn(),
  searchEstimate: vi.fn(),
  factors: vi.fn(),
  capabilities: vi.fn(),
  experiments: vi.fn(),
  models: vi.fn(),
  modelDetail: vi.fn(),
  modelBacktest: vi.fn(),
  modelPredictions: vi.fn(),
  strategies: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  api: {
    quantFactors: mocks.factors,
    quantMLCapabilities: mocks.capabilities,
    quantExperiments: mocks.experiments,
    quantModels: mocks.models,
    quantModelDetail: mocks.modelDetail,
    quantModelBacktest: mocks.modelBacktest,
    quantModelPredictions: mocks.modelPredictions,
    quantPredictionDates: vi.fn(),
    quantStrategies: mocks.strategies,
    quantTrain: mocks.train,
    quantSearch: mocks.search,
    quantSearchEstimate: mocks.searchEstimate,
    quantPublishModel: vi.fn(),
    quantArchiveModel: vi.fn(),
    quantGeneratePredictions: vi.fn(),
    quantCancelExperiment: vi.fn(),
    quantRerunExperiment: vi.fn(),
    quantDeleteExperiment: vi.fn(),
    quantSaveStrategy: vi.fn(),
    quantDeleteStrategy: vi.fn(),
    quantOptimizePortfolio: vi.fn(),
  },
}))

const factors = [
  {
    id: 'momentum_20d', name: '20日动量', description: '月度动量', family: '动量', version: 'factor-v1',
    authoring_type: 'builtin', asset_types: ['stock', 'etf'], trusted: true, readonly: true, point_in_time: true,
  },
  {
    id: 'annual_vol_20d', name: '20日波动率', description: '年化波动', family: '波动率', version: 'factor-v1',
    authoring_type: 'builtin', asset_types: ['stock', 'etf'], trusted: true, readonly: true, point_in_time: true,
  },
  {
    id: 'rsi_14', name: 'RSI(14)', description: '相对强弱', family: '动量', version: 'factor-v1',
    authoring_type: 'builtin', asset_types: ['stock', 'etf'], trusted: true, readonly: true, point_in_time: true,
  },
  {
    id: 'turnover_rate', name: '换手率', description: '当日换手', family: '量价', version: 'factor-v1',
    authoring_type: 'builtin', asset_types: ['stock', 'etf'], trusted: true, readonly: true, point_in_time: true,
  },
]

function renderQuant() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}><Quant /></QueryClientProvider>)
}

describe('Quant workspace', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.factors.mockResolvedValue({ factors })
    mocks.capabilities.mockResolvedValue({
      gpu: { available: true, name: 'NVIDIA GeForce RTX 5060 Ti', memory_mb: 8151 },
      cpu_threads: 18,
      algorithms: {
        elastic_net: { installed: true, version: '1.8.0', gpu_backend: null, gpu_candidate: false },
        lightgbm: { installed: true, version: '4.6.0', gpu_backend: 'opencl', gpu_candidate: true },
        xgboost: { installed: true, version: '3.3.0', gpu_backend: 'cuda', gpu_candidate: true },
      },
      sklearn: { installed: true, version: '1.8.0' },
      optuna: { installed: true, version: '4.9.0' },
      joblib: { installed: true, version: '1.5.3' },
    })
    mocks.experiments.mockResolvedValue({ experiments: [] })
    mocks.models.mockResolvedValue({ models: [] })
    mocks.modelDetail.mockResolvedValue(null)
    mocks.modelBacktest.mockResolvedValue({ run_id: 'backtest-1', status: 'queued' })
    mocks.modelPredictions.mockResolvedValue({ predictions: [], total: 0, date: null, summary: null })
    mocks.strategies.mockResolvedValue({ strategies: [] })
    mocks.train.mockResolvedValue({ run_id: 'run-1', status: 'queued' })
    mocks.search.mockResolvedValue({ run_id: 'search-1', status: 'queued' })
    mocks.searchEstimate.mockResolvedValue({
      estimated_rows: 100_000, factor_count: 4, outer_folds: 3,
      search_trials_per_window: 72, estimated_model_fits: 868,
      estimated_hours: 3, warnings: [],
    })
  })

  it('shows local GPU capabilities and registered factors', async () => {
    const user = userEvent.setup()
    renderQuant()
    expect(await screen.findByText(/RTX 5060 Ti/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '因子库' }))
    expect(await screen.findByText('20日动量')).toBeInTheDocument()
    expect(screen.getAllByText('factor-v1')).toHaveLength(4)
  })

  it('submits the leakage-safe walk-forward defaults', async () => {
    const user = userEvent.setup()
    renderQuant()
    await user.click(await screen.findByRole('button', { name: '手工训练' }))
    await user.click(await screen.findByRole('button', { name: /开始 Walk-forward 训练/ }))
    await waitFor(() => expect(mocks.train).toHaveBeenCalledOnce())
    const spec = mocks.train.mock.calls[0][0]
    expect(spec.algorithm).toBe('xgboost')
    expect(spec.target).toMatchObject({ horizon: 5, benchmark_symbol: '000300.SH' })
    expect(spec.walk_forward).toEqual({ train_days: 756, validation_days: 126, test_days: 126, step_days: 126 })
    expect(spec.features).toEqual(['momentum_20d', 'annual_vol_20d', 'rsi_14', 'turnover_rate'])
  })

  it('submits exact factor versions and nested search defaults', async () => {
    const user = userEvent.setup()
    renderQuant()
    await user.click(await screen.findByRole('button', { name: '估算资源' }))
    await waitFor(() => expect(mocks.searchEstimate).toHaveBeenCalledOnce())
    await user.click(screen.getByRole('button', { name: /开始智能训练/ }))
    await waitFor(() => expect(mocks.search).toHaveBeenCalledOnce())
    const spec = mocks.search.mock.calls[0][0]
    expect(spec.factor_pool).toEqual(factors.map(item => ({ id: item.id, version: item.version })))
    expect(spec.algorithms).toEqual(['elastic_net', 'lightgbm', 'xgboost'])
    expect(spec.budget).toBe('standard')
    expect(spec.inner_folds).toBe(3)
    expect(spec.inner_validation_days).toBe(63)
    expect(spec.walk_forward).toEqual({ train_days: 756, validation_days: 126, test_days: 126, step_days: 126 })
  })

  it('opens model diagnostics and starts an OOS backtest', async () => {
    const model = {
      version: 'model-v1', model_id: 'model_1', name: 'A股收益模型', algorithm: 'xgboost',
      status: 'published', created_at: '2026-07-16T00:00:00', published_at: '2026-07-16T01:00:00',
      spec: {
        id: 'model_1', name: 'A股收益模型', algorithm: 'xgboost', asset_type: 'stock', symbols: null,
        features: ['momentum_20d'], start: '2021-01-01', end: '2026-07-16',
        target: { horizon: 5, benchmark_mode: 'index', benchmark_symbol: '000300.SH' },
        walk_forward: { train_days: 756, validation_days: 126, test_days: 126, step_days: 126 },
        tuning: { enabled: false, max_trials: 20 }, device: 'auto', params: {}, seed: 42,
        universe_filters: {},
      },
      metrics: { rank_ic: 0.05, icir: 0.5 },
      training: { actual_devices: ['cuda'], library_versions: ['3.3.0'], training_seconds: 7, warnings: [] },
      diagnostic: {
        grade: 'candidate', publish_warning: false, warnings: [],
        dimensions: {
          data: { status: 'green', reason: '2 折 / 260 个 OOS 交易日' },
          statistics: { status: 'green', reason: 'Rank IC 0.05' },
          stability: { status: 'green', reason: '正 IC 折 100%' },
          economics: { status: 'yellow', reason: '待完成 OOS 组合回测' },
        },
      },
      latest_backtest: null, latest_prediction: null,
    }
    mocks.models.mockResolvedValue({ models: [model] })
    mocks.modelPredictions.mockResolvedValue({
      predictions: [{
        symbol: '000001.SZ', name: '平安银行', date: '2026-07-16', model_version: 'model-v1',
        prediction: 0.018, rank: 0.998, feature_coverage: 1,
      }],
      total: 1,
      date: '2026-07-16',
      summary: {
        date: '2026-07-16', rows: 1, coverage: 1, prediction_min: 0.018,
        prediction_max: 0.018, prediction_mean: 0.018, psi: 0.08, warnings: [],
      },
    })
    mocks.modelDetail.mockResolvedValue({
      ...model, training_run: { result: { metrics: { daily_ic: [] }, folds: [], feature_importance: {} } },
      backtests: [], latest_backtest: null,
      prediction_dates: [{
        date: '2026-07-16', rows: 1, coverage: 1, prediction_min: 0.018,
        prediction_max: 0.018, prediction_mean: 0.018, psi: 0.08, warnings: [],
      }],
    })
    const user = userEvent.setup()
    renderQuant()
    const modelCenterButtons = await screen.findAllByRole('button', { name: '模型中心' })
    await user.click(modelCenterButtons.at(-1)!)
    expect((await screen.findAllByText('A股收益模型')).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: '组合回测' }))
    await user.click(await screen.findByRole('button', { name: /运行严格 OOS 回测/ }))
    await waitFor(() => expect(mocks.modelBacktest).toHaveBeenCalledOnce())
    expect(mocks.modelBacktest.mock.calls[0][1]).toMatchObject({
      model_version: 'model-v1', top_n: 10, rebalance_days: 5, weighting: 'equal',
    })
    await user.click(screen.getByRole('button', { name: '盘后预测' }))
    expect(await screen.findByText('平安银行')).toBeInTheDocument()
    expect(mocks.modelPredictions).toHaveBeenCalledWith('model-v1', '2026-07-16', '', 10_000)
  })

  it('shows the AutoML factor funnel and champion evidence', async () => {
    mocks.experiments.mockResolvedValue({ experiments: [{
      run_id: 'search-run', kind: 'ml_search', status: 'completed',
      created_at: '2026-07-16T00:00:00', updated_at: '2026-07-16T01:00:00',
      progress: 1, message: '智能训练完成', error: null, warnings: [],
      spec: { name: '多因子冠军搜索' },
      result: {
        champion: { algorithm: 'elastic_net', features: ['momentum_20d'], selection_score: 0.72 },
        factor_funnel: { submitted: 4, quality_passed: 3, shortlisted: 2, selected: 1 },
        metrics: { rank_ic: 0.04, icir: 0.55 }, warnings: [],
        feature_selection_frequency: { momentum_20d: 1 },
        candidate_leaderboard: [{
          trial: 0, algorithm: 'elastic_net', features: ['momentum_20d'], score: 0.72,
          metrics: { rank_ic: 0.04, icir: 0.55 },
          economic: { annual_excess_vs_index: 0.08, annual_excess_vs_universe: 0.06, sharpe: 0.9 },
        }],
        factor_quality: [{
          factor_id: 'momentum_20d', status: 'accepted', coverage: 0.98,
          rank_ic: 0.04, icir: 0.55, reason: null,
        }],
      },
    }] })
    const user = userEvent.setup()
    renderQuant()
    const experimentButtons = await screen.findAllByRole('button', { name: '实验记录' })
    await user.click(experimentButtons.at(-1)!)
    const resultButtons = await screen.findAllByRole('button', { name: '查看结果' })
    await user.click(resultButtons.at(-1)!)
    expect(await screen.findByText('因子漏斗明细')).toBeInTheDocument()
    expect(screen.getAllByText('elastic_net').length).toBeGreaterThan(0)
    expect(screen.getByText('通过质量筛选')).toBeInTheDocument()
  })
})
