import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { Financials } from './Financials'

const mocks = vi.hoisted(() => ({
  capabilities: vi.fn(),
  financialStatus: vi.fn(),
  financialSync: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  api: {
    capabilities: mocks.capabilities,
    financialStatus: mocks.financialStatus,
    financialSync: mocks.financialSync,
  },
}))

vi.mock('@/components/financials/StockFinancialSearch', () => ({
  StockFinancialSearch: () => <div>本地财务搜索</div>,
}))
vi.mock('@/components/financials/StockFinancialDetail', () => ({
  StockFinancialDetail: () => <div>财务详情</div>,
}))
vi.mock('@/components/financials/ReportHistoryPanel', () => ({
  ReportHistoryPanel: () => <div>历史报告</div>,
}))
vi.mock('@/components/LastStockChip', () => ({
  LastStockChip: () => null,
}))
vi.mock('@/lib/useLastStock', () => ({
  useLastStock: () => ({ last: null, remember: vi.fn() }),
}))

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <Financials />
    </QueryClientProvider>,
  )
}

describe('Financials local data access', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.capabilities.mockResolvedValue({ label: 'Pro', capabilities: {} })
    mocks.financialStatus.mockResolvedValue({
      available: true,
      can_sync: false,
      syncing: false,
      last_sync: {},
      tables: {
        metrics: { rows: 5833, symbols: 5833, updated_at: '2026-07-19T15:39:14' },
        income: { rows: 0, symbols: 0, updated_at: null },
        balance_sheet: { rows: 0, symbols: 0, updated_at: null },
        cash_flow: { rows: 0, symbols: 0, updated_at: null },
      },
    })
  })

  it('shows local metrics instead of the Expert lock screen', async () => {
    renderPage()

    expect(await screen.findByText('本地只读')).toBeInTheDocument()
    expect(await screen.findByText(/正在展示本地已有财务数据/)).toBeInTheDocument()
    expect(await screen.findByText('本地财务搜索')).toBeInTheDocument()
    expect(screen.queryByText('需要 Expert 套餐')).not.toBeInTheDocument()
  })
})
