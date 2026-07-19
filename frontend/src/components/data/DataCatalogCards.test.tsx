import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DataDimensionStatus } from '@/lib/api'
import { NewsDataCard } from './NewsDataCard'
import { PersistentAssetSections } from './PersistentAssetSections'

afterEach(cleanup)

function dimension(
  overrides: Partial<DataDimensionStatus> = {},
): DataDimensionStatus {
  return {
    id: 'finance_news',
    label: '财联社快讯',
    category: 'business',
    state: 'ready',
    records: 12,
    files: 3,
    parquet_files: 2,
    size_mb: 0.79,
    earliest_at: '2026-07-11',
    latest_at: '2026-07-18',
    last_modified_at: '2026-07-18T10:00:00+08:00',
    sensitive: false,
    children: [],
    sync: {
      mode: 'scheduled',
      last_success_at: '2026-07-18T10:00:00+08:00',
      next_run_at: '2026-07-18T10:01:00+08:00',
      error: null,
    },
    ...overrides,
  }
}

describe('data catalog cards', () => {
  it('shows news totals and exposes refresh, fields, and news entry', () => {
    const onRefresh = vi.fn()
    const onShowFields = vi.fn()
    render(
      <MemoryRouter>
        <NewsDataCard
          dimension={dimension()}
          loading={false}
          refreshing={false}
          onRefresh={onRefresh}
          onShowFields={onShowFields}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('2026-07-11 至 2026-07-18')).toBeInTheDocument()
    expect(screen.getByTitle('打开快讯页面')).toHaveAttribute('href', '/news')
    fireEvent.click(screen.getByTitle('刷新财联社快讯'))
    fireEvent.click(screen.getByTitle('查看字段说明'))
    expect(onRefresh).toHaveBeenCalledOnce()
    expect(onShowFields).toHaveBeenCalledOnce()
  })

  it('keeps old news totals visible when refresh fails', () => {
    render(
      <MemoryRouter>
        <NewsDataCard
          dimension={dimension({ state: 'error' })}
          loading={false}
          refreshing={false}
          refreshError="409: 财联社新闻正在同步中"
          onRefresh={() => undefined}
          onShowFields={() => undefined}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('同步异常')).toBeInTheDocument()
    expect(screen.getByText('409: 财联社新闻正在同步中')).toBeInTheDocument()
  })

  it('expands research and system asset summaries', () => {
    const research = dimension({
      id: 'quant_research',
      label: '量化研究',
      category: 'research',
      children: [
        dimension({
          id: 'quant_models',
          label: '模型',
          category: 'research',
          records: null,
          children: [],
        }),
      ],
    })
    const system = dimension({
      id: 'configuration',
      label: '配置与凭据',
      category: 'system',
      sensitive: true,
      children: [],
    })
    render(<PersistentAssetSections dimensions={[research, system]} />)

    fireEvent.click(screen.getByText('用户与研究'))
    fireEvent.click(screen.getByText('量化研究'))
    expect(screen.getByText('模型')).toBeInTheDocument()
    fireEvent.click(screen.getByText('系统资产'))
    expect(screen.getByText('配置与凭据')).toBeInTheDocument()
    expect(screen.getByTitle('敏感资产仅展示汇总')).toBeInTheDocument()
  })
})
