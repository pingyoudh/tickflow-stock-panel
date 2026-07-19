import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { News } from './News'

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  refresh: vi.fn(),
  toast: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  api: {
    financeNewsList: mocks.list,
    financeNewsRefresh: mocks.refresh,
  },
}))

vi.mock('@/components/Toast', () => ({
  toast: mocks.toast,
}))

const firstItem = {
  news_id: '1',
  source: 'cls' as const,
  url: 'https://api3.cls.cn/share/article/1?os=web&sv=8.4.6&app=CailianpressWeb',
  title: '',
  content: `机器人板块出现异动。${'后续正文'.repeat(80)}`,
  published_at: '2026-07-18T10:30:00+08:00',
  modified_at: '2026-07-18T10:30:00+08:00',
  level: 'A',
  recommend: true,
  subjects: [{ subject_id: 7, subject_name: '机器人' }],
  stocks: [{ stock_code: '600000.SH', stock_name: '浦发银行' }],
}

const syncStatus = {
  syncing: false,
  backfill_completed: false,
  last_success_at: '2026-07-18T10:31:00+08:00',
  last_error: '上次请求超时',
  latest_published_at: '2026-07-18T10:30:00+08:00',
}

function renderNews() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <News />
    </QueryClientProvider>,
  )
}

describe('News page', () => {
  afterEach(cleanup)

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.list.mockImplementation((_limit: number, cursor?: string) => Promise.resolve(
      cursor
        ? {
            items: [{ ...firstItem, news_id: '2', title: '第二页快讯', content: '第二页正文' }],
            next_cursor: null,
            has_more: false,
            sync_status: { ...syncStatus, backfill_completed: true, last_error: null },
          }
        : {
            items: [firstItem],
            next_cursor: 'next-page',
            has_more: true,
            sync_status: syncStatus,
          },
    ))
    mocks.refresh.mockResolvedValue({
      fetched: 20,
      inserted: 1,
      updated: 0,
      latest_published_at: firstItem.published_at,
      synced_at: '2026-07-18T10:32:00+08:00',
    })
  })

  it('renders fallback title, sync states, expansion and cursor pagination', async () => {
    const user = userEvent.setup()
    renderNews()

    expect(await screen.findByRole('heading', { name: '机器人板块出现异动' })).toBeInTheDocument()
    expect(screen.getByText(/正在回补最近 7 天快讯/)).toBeInTheDocument()
    expect(screen.getByText(/最近同步失败：上次请求超时/)).toBeInTheDocument()
    expect(screen.getByText('浦发银行')).toBeInTheDocument()
    expect(screen.getByText('600000.SH')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '机器人板块出现异动' })).toHaveAttribute(
      'href',
      firstItem.url,
    )

    await user.click(screen.getByRole('button', { name: '展开' }))
    expect(screen.getByRole('button', { name: '收起' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '加载更多' }))
    expect(await screen.findByRole('heading', { name: '第二页快讯' })).toBeInTheDocument()
    expect(mocks.list).toHaveBeenCalledWith(50, 'next-page')
  })

  it('manually refreshes and invalidates the news list', async () => {
    const user = userEvent.setup()
    renderNews()
    await screen.findByRole('heading', { name: '机器人板块出现异动' })

    await user.click(screen.getByRole('button', { name: '刷新' }))

    await waitFor(() => expect(mocks.refresh).toHaveBeenCalledTimes(1))
    expect(mocks.toast).toHaveBeenCalledWith('同步完成，新增或更新 1 条快讯', 'success')
  })

  it('retains stored news when a manual refresh conflicts or fails', async () => {
    mocks.refresh.mockRejectedValue(new Error('财联社新闻正在同步中'))
    const user = userEvent.setup()
    renderNews()
    await screen.findByRole('heading', { name: '机器人板块出现异动' })

    await user.click(screen.getByRole('button', { name: '刷新' }))

    await waitFor(() => expect(mocks.refresh).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('heading', { name: '机器人板块出现异动' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '刷新' })).toBeEnabled()
  })

  it('shows an empty state when no news has been stored', async () => {
    mocks.list.mockResolvedValue({
      items: [],
      next_cursor: null,
      has_more: false,
      sync_status: { ...syncStatus, last_error: null },
    })
    renderNews()

    expect(await screen.findByText('暂无快讯')).toBeInTheDocument()
    expect(screen.getByText('后台正在获取财联社数据')).toBeInTheDocument()
  })
})
