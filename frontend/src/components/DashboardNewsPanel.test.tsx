import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { FinanceNewsItem, FinanceNewsPage } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { DashboardNewsPanel } from './DashboardNewsPanel'

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  activateVoice: vi.fn(),
  speakVoiceText: vi.fn(() => true),
  stopVoice: vi.fn(),
}))

vi.mock('@/lib/api', () => ({
  api: { financeNewsList: mocks.list },
}))

vi.mock('@/lib/voiceBroadcast', () => ({
  activateVoice: mocks.activateVoice,
  isVoiceSupported: () => true,
  speakVoiceText: mocks.speakVoiceText,
  stopVoice: mocks.stopVoice,
}))

const normalItem: FinanceNewsItem = {
  news_id: '100',
  source: 'cls',
  url: 'https://api3.cls.cn/share/article/100?os=web&sv=8.4.6&app=CailianpressWeb',
  title: '',
  content: '普通快讯正文第一句。后续内容',
  published_at: '2026-07-18T10:30:00+08:00',
  modified_at: '2026-07-18T10:30:00+08:00',
  level: 'C',
  recommend: false,
  subjects: [{ subject_id: 1, subject_name: '机器人' }],
  stocks: [],
}

const importantItem: FinanceNewsItem = {
  ...normalItem,
  news_id: '101',
  title: '已有重点快讯',
  level: 'B',
  recommend: true,
  stocks: [{ stock_code: '600000.SH', stock_name: '浦发银行' }],
}

function page(items: FinanceNewsItem[]): FinanceNewsPage {
  return {
    items,
    next_cursor: null,
    has_more: false,
    sync_status: {
      syncing: false,
      backfill_completed: true,
      last_success_at: '2026-07-18T10:31:00+08:00',
      last_error: null,
      latest_published_at: items[0]?.published_at ?? null,
    },
  }
}

function renderPanel(initialItems = [normalItem, importantItem]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  mocks.list.mockResolvedValue(page(initialItems))
  const view = render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <DashboardNewsPanel />
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { ...view, client }
}

describe('DashboardNewsPanel', () => {
  afterEach(cleanup)

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('renders latest news with fallback titles and source-driven importance marks', async () => {
    renderPanel()

    expect(await screen.findByText('普通快讯正文第一句')).toBeInTheDocument()
    expect(screen.getByText('已有重点快讯')).toBeInTheDocument()
    expect(screen.getAllByText('重点')).toHaveLength(1)
    expect(screen.getAllByText('机器人')).toHaveLength(2)
    expect(screen.getByText('浦发银行')).toBeInTheDocument()
    expect(screen.getByText('600000.SH')).toBeInTheDocument()
    expect(screen.getByRole('link', {
      name: '打开财联社新闻：普通快讯正文第一句',
    })).toHaveAttribute('href', normalItem.url)
    expect(mocks.list).toHaveBeenCalledWith(50)
  })

  it('speaks only newly arrived important news once after the user enables voice', async () => {
    const user = userEvent.setup()
    const { client } = renderPanel()
    await screen.findByText('已有重点快讯')

    await user.click(screen.getByRole('button', { name: '开启重点快讯播报' }))
    expect(mocks.activateVoice).toHaveBeenCalledTimes(1)
    expect(mocks.speakVoiceText).not.toHaveBeenCalled()

    const arrived: FinanceNewsItem = {
      ...importantItem,
      news_id: '102',
      title: '刚刚到达的重点快讯',
      published_at: '2026-07-18T10:32:00+08:00',
      modified_at: '2026-07-18T10:32:00+08:00',
    }
    act(() => {
      client.setQueryData(QK.financeNewsDashboard, page([arrived, normalItem, importantItem]))
    })

    await waitFor(() => {
      expect(mocks.speakVoiceText).toHaveBeenCalledWith('重点快讯。刚刚到达的重点快讯')
    })

    act(() => {
      client.setQueryData(QK.financeNewsDashboard, page([arrived, normalItem, importantItem]))
    })
    await waitFor(() => expect(mocks.speakVoiceText).toHaveBeenCalledTimes(1))
    expect(localStorage.getItem('dashboard-news-spoken-ids')).toContain('cls:102')
  })

  it('does not speak newly arrived non-important news', async () => {
    const user = userEvent.setup()
    const { client } = renderPanel()
    await screen.findByText('已有重点快讯')
    await user.click(screen.getByRole('button', { name: '开启重点快讯播报' }))

    const arrived = {
      ...normalItem,
      news_id: '103',
      title: '新到普通快讯',
    }
    act(() => {
      client.setQueryData(QK.financeNewsDashboard, page([arrived, normalItem, importantItem]))
    })

    expect(await screen.findByText('新到普通快讯')).toBeInTheDocument()
    expect(mocks.speakVoiceText).not.toHaveBeenCalled()
  })

  it('pins the latest same-day important item when it falls outside the compact list', async () => {
    const recent = Array.from({ length: 12 }, (_, index): FinanceNewsItem => ({
      ...normalItem,
      news_id: `recent-${index}`,
      title: `普通快讯 ${index + 1}`,
      published_at: `2026-07-18T${String(12 - Math.floor(index / 2)).padStart(2, '0')}:${index % 2 ? '15' : '45'}:00+08:00`,
    }))
    const olderImportant = {
      ...importantItem,
      news_id: 'older-important',
      title: '今日较早重点快讯',
      published_at: '2026-07-18T06:30:00+08:00',
    }
    const olderRelated = {
      ...normalItem,
      news_id: 'older-related',
      title: '今日较早关联股快讯',
      published_at: '2026-07-18T07:30:00+08:00',
      stocks: [{ stock_code: '600699.SH', stock_name: '均胜电子' }],
    }
    renderPanel([...recent, olderImportant, olderRelated])

    expect(await screen.findByText('今日较早重点快讯')).toBeInTheDocument()
    expect(screen.getByText('今日较早关联股快讯')).toBeInTheDocument()
    expect(screen.getByText('均胜电子')).toBeInTheDocument()
    expect(screen.getByText('600699.SH')).toBeInTheDocument()
    expect(screen.getAllByText('重点')).toHaveLength(1)
  })
})
