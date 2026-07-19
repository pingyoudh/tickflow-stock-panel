import { ExternalLink, Loader2, Newspaper, RefreshCw, Table2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { DataDimensionStatus } from '@/lib/api'
import { formatNumber } from '@/lib/format'

function formatTime(value: string | null | undefined) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function stateText(dimension: DataDimensionStatus | undefined) {
  if (!dimension) return '未初始化'
  if (dimension.state === 'error') return '同步异常'
  if (dimension.state === 'syncing') {
    return dimension.sync?.last_success_at ? '同步中' : '首次回补'
  }
  if (dimension.state === 'empty') return '等待回补'
  return '已同步'
}

export function NewsDataCard({
  dimension,
  loading,
  refreshing,
  refreshError,
  onRefresh,
  onShowFields,
}: {
  dimension?: DataDimensionStatus
  loading: boolean
  refreshing: boolean
  refreshError?: string | null
  onRefresh: () => void
  onShowFields: () => void
}) {
  const error = refreshError || dimension?.sync?.error
  const syncing = refreshing || dimension?.state === 'syncing'
  const tone = error
    ? 'text-danger bg-danger/8'
    : syncing
      ? 'text-warning bg-warning/8'
      : 'text-accent bg-accent/8'

  return (
    <div className="min-h-[224px] rounded-card border border-border bg-surface flex flex-col">
      <div className="flex items-start justify-between px-4 pt-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Newspaper className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium text-foreground">财联社快讯</h3>
          </div>
          <p className="mt-1 text-[10px] text-muted">全天增量 · 最近 7 天回补</p>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onShowFields}
            className="p-1 rounded hover:bg-elevated text-secondary hover:text-accent transition-colors"
            title="查看字段说明"
          >
            <Table2 className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={onRefresh}
            disabled={refreshing}
            className="p-1 rounded hover:bg-elevated text-secondary hover:text-accent transition-colors disabled:opacity-50"
            title="刷新财联社快讯"
          >
            {refreshing
              ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
              : <RefreshCw className="h-3.5 w-3.5" />}
          </button>
          <Link
            to="/news"
            className="p-1 rounded hover:bg-elevated text-secondary hover:text-accent transition-colors"
            title="打开快讯页面"
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>

      <div className="px-4 pt-3">
        {loading ? (
          <div className="h-8 w-20 animate-pulse rounded bg-elevated" />
        ) : (
          <>
            <div className="font-mono text-2xl font-bold tabular-nums text-foreground">
              {dimension?.records == null ? '—' : formatNumber(dimension.records)}
            </div>
            <div className="mt-0.5 text-[11px] text-muted">条快讯</div>
          </>
        )}
      </div>

      <div className="mt-auto border-t border-border px-4 py-3 space-y-1.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className={`rounded px-1.5 py-px font-medium ${tone}`}>
            {stateText(dimension)}
          </span>
          <span className="font-mono text-muted">
            {dimension ? `${dimension.files} 文件 · ${dimension.size_mb.toFixed(2)} MB` : '—'}
          </span>
        </div>
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-muted">范围</span>
          <span className="font-mono text-secondary">
            {dimension?.earliest_at && dimension.latest_at
              ? `${dimension.earliest_at} 至 ${dimension.latest_at}`
              : '—'}
          </span>
        </div>
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-muted">最近同步</span>
          <span className="font-mono text-secondary">
            {formatTime(dimension?.sync?.last_success_at)}
          </span>
        </div>
        {error && (
          <p className="truncate text-[10px] text-danger" title={error}>
            {error}
          </p>
        )}
      </div>
    </div>
  )
}
