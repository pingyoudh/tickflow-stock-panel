import { useMemo, useState } from 'react'
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Clock3,
  ExternalLink,
  Loader2,
  Newspaper,
  RefreshCw,
  Sparkles,
} from 'lucide-react'

import { PageHeader } from '@/components/PageHeader'
import { toast } from '@/components/Toast'
import { MarkdownRenderer } from '@/components/financials/MarkdownRenderer'
import { api, type FinanceNewsItem } from '@/lib/api'
import { cn } from '@/lib/cn'
import { financeNewsTitle } from '@/lib/financeNews'
import { QK } from '@/lib/queryKeys'

const PAGE_SIZE = 50
type SummaryPhase = 'idle' | 'loading' | 'streaming' | 'done' | 'error'

function formatPublishedAt(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return { date: '--', time: '--' }
  return {
    date: new Intl.DateTimeFormat('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      timeZone: 'Asia/Shanghai',
    }).format(date),
    time: new Intl.DateTimeFormat('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: 'Asia/Shanghai',
    }).format(date),
  }
}

function formatSyncTime(value: string | null | undefined) {
  if (!value) return '尚未完成同步'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '同步时间未知'
  return `最近同步 ${new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Shanghai',
  }).format(date)}`
}

function levelClass(level: string) {
  if (level === 'A') return 'border-danger/30 bg-danger/10 text-danger'
  if (level === 'B') return 'border-warning/30 bg-warning/10 text-warning'
  return 'border-border bg-elevated text-muted'
}

function NewsRow({ item }: { item: FinanceNewsItem }) {
  const [expanded, setExpanded] = useState(false)
  const published = formatPublishedAt(item.published_at)
  const title = financeNewsTitle(item)
  const isLong = item.content.length > 220
  const subjects = item.subjects.filter(subject => subject.subject_name)
  const stocks = item.stocks.filter(stock => stock.stock_code || stock.stock_name)

  return (
    <article className="grid min-w-0 gap-2 border-b border-border/70 px-3 py-3 last:border-b-0 sm:grid-cols-[5.5rem_minmax(0,1fr)] sm:gap-4 sm:px-4">
      <div className="flex items-baseline gap-2 text-xs sm:block">
        <div className="font-mono text-sm font-semibold text-foreground">{published.time}</div>
        <div className="mt-0.5 text-[10px] text-muted">{published.date}</div>
      </div>

      <div className="min-w-0">
        <div className="flex min-w-0 flex-wrap items-start gap-1.5">
          {item.level && (
            <span className={cn('shrink-0 rounded-sm border px-1.5 py-0.5 text-[10px] font-semibold', levelClass(item.level))}>
              {item.level}
            </span>
          )}
          {item.recommend && (
            <span className="inline-flex shrink-0 items-center gap-1 rounded-sm border border-accent/30 bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">
              <Sparkles className="h-2.5 w-2.5" />推荐
            </span>
          )}
          <h2 className="min-w-0 flex-1 text-sm font-semibold leading-5 text-foreground">
            <a
              href={item.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-start gap-1 transition-colors hover:text-accent"
            >
              <span>{title}</span>
              <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" />
            </a>
          </h2>
        </div>

        {item.content && (
          <p className={cn(
            'mt-1.5 whitespace-pre-wrap break-words text-xs leading-5 text-secondary',
            !expanded && isLong && 'line-clamp-3',
          )}>
            {item.content}
          </p>
        )}

        <div className="mt-2 flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted">
          <span>财联社</span>
          {subjects.length > 0 && (
            <span className="min-w-0 truncate">题材：{subjects.map(subject => subject.subject_name).join(' · ')}</span>
          )}
          {stocks.length > 0 && (
            <span className="flex min-w-0 flex-wrap items-center gap-1">
              <span>个股：</span>
              {stocks.map(stock => (
                <span
                  key={`${stock.stock_code}:${stock.stock_name}`}
                  className="inline-flex items-center gap-1 rounded-sm bg-accent/10 px-1.5 py-0.5 text-accent"
                >
                  {stock.stock_name || stock.stock_code}
                  {stock.stock_name && stock.stock_code && (
                    <span className="font-mono text-[9px] text-accent/70">{stock.stock_code}</span>
                  )}
                </span>
              ))}
            </span>
          )}
          {isLong && (
            <button
              type="button"
              onClick={() => setExpanded(value => !value)}
              className="ml-auto inline-flex shrink-0 items-center gap-0.5 text-accent transition-colors hover:text-accent/80"
            >
              {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              {expanded ? '收起' : '展开'}
            </button>
          )}
        </div>
      </div>
    </article>
  )
}

export function News() {
  const queryClient = useQueryClient()
  const [refreshing, setRefreshing] = useState(false)
  const [summaryPhase, setSummaryPhase] = useState<SummaryPhase>('idle')
  const [summaryContent, setSummaryContent] = useState('')
  const [summaryError, setSummaryError] = useState('')
  const [summaryProgress, setSummaryProgress] = useState('')
  const [summaryMeta, setSummaryMeta] = useState<{
    unique_count?: number
    latest_published_at?: string | null
    generated_at?: string
    market_snapshot_at?: string | null
    market_ready?: boolean
    tail_window?: boolean
    eligible_count?: number
    market_warnings?: string[]
  } | null>(null)
  const newsQuery = useInfiniteQuery({
    queryKey: QK.financeNews,
    queryFn: ({ pageParam }) => api.financeNewsList(
      PAGE_SIZE,
      typeof pageParam === 'string' ? pageParam : undefined,
    ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: page => page.has_more ? page.next_cursor ?? undefined : undefined,
    refetchInterval: () => document.visibilityState === 'visible' ? 60_000 : false,
  })
  const refreshMutation = useMutation({
    mutationFn: api.financeNewsRefresh,
    onMutate: () => setRefreshing(true),
    onSuccess: result => {
      const changed = result.inserted + result.updated
      toast(changed > 0 ? `同步完成，新增或更新 ${changed} 条快讯` : '已是最新快讯', 'success')
      queryClient.invalidateQueries({ queryKey: QK.financeNews })
      queryClient.invalidateQueries({ queryKey: ['finance-news', 'daily-summary'] })
    },
    onSettled: () => setRefreshing(false),
  })
  const summaryQuery = useQuery({
    queryKey: ['finance-news', 'daily-summary'],
    queryFn: () => api.financeNewsDailySummary(),
    refetchInterval: () => document.visibilityState === 'visible' ? 60_000 : false,
  })

  const startSummary = async () => {
    if (summaryPhase === 'loading' || summaryPhase === 'streaming') return
    const force = Boolean(summaryQuery.data?.summary)
    setSummaryPhase('loading')
    setSummaryContent('')
    setSummaryError('')
    setSummaryProgress('正在准备当日新闻')
    setSummaryMeta(null)
    let content = ''
    try {
      for await (const event of api.financeNewsAnalyzeStream(force)) {
        if (event.type === 'meta') {
          setSummaryMeta({
            unique_count: event.unique_count,
            latest_published_at: event.latest_published_at,
            market_snapshot_at: event.market_snapshot_at,
            market_ready: event.market_ready,
            tail_window: event.tail_window,
            eligible_count: event.eligible_count,
            market_warnings: event.market_warnings,
          })
          setSummaryProgress(event.cache_hit ? '正在读取已有报告' : '正在合并新闻与盘面数据')
        } else if (event.type === 'progress') {
          setSummaryProgress(event.message ?? '正在生成总结')
        } else if (event.type === 'delta' && event.content) {
          content += event.content
          setSummaryContent(content)
          setSummaryPhase('streaming')
        } else if (event.type === 'error') {
          setSummaryError(event.message ?? '新闻总结失败')
          setSummaryPhase('error')
          return
        } else if (event.type === 'done') {
          setSummaryMeta(previous => ({
            ...(previous ?? {}),
            unique_count: event.unique_count ?? previous?.unique_count,
            latest_published_at: event.latest_published_at ?? previous?.latest_published_at,
            generated_at: event.generated_at,
            market_snapshot_at: event.market_snapshot_at ?? previous?.market_snapshot_at,
            market_ready: event.market_ready ?? previous?.market_ready,
            tail_window: event.tail_window ?? previous?.tail_window,
            eligible_count: event.eligible_count ?? previous?.eligible_count,
            market_warnings: event.market_warnings ?? previous?.market_warnings,
          }))
          setSummaryPhase('done')
          queryClient.invalidateQueries({ queryKey: ['finance-news', 'daily-summary'] })
        }
      }
      if (content) setSummaryPhase('done')
    } catch (error) {
      setSummaryError(error instanceof Error ? error.message : '新闻总结失败')
      setSummaryPhase('error')
    }
  }

  const pages = newsQuery.data?.pages ?? []
  const items = useMemo(() => pages.flatMap(page => page.items), [pages])
  const syncStatus = pages[0]?.sync_status
  const syncing = refreshing || refreshMutation.isPending || syncStatus?.syncing
  const storedSummary = summaryQuery.data?.summary
  const summaryRunning = summaryPhase === 'loading' || summaryPhase === 'streaming'
  const visibleSummary = summaryPhase === 'idle'
    ? storedSummary?.content ?? ''
    : summaryContent || storedSummary?.content || ''
  const summaryCount = summaryMeta?.unique_count
    ?? storedSummary?.unique_count
    ?? summaryQuery.data?.current_unique_count
    ?? 0
  const summaryLatestAt = summaryMeta?.latest_published_at
    ?? storedSummary?.latest_published_at
  const summaryGeneratedAt = summaryMeta?.generated_at ?? storedSummary?.generated_at
  const marketSnapshotAt = summaryMeta?.market_snapshot_at
    ?? storedSummary?.market_snapshot_at
    ?? summaryQuery.data?.market?.snapshot_at
  const marketReady = summaryMeta?.market_ready
    ?? storedSummary?.market_ready
    ?? summaryQuery.data?.market?.ready
    ?? false
  const tailWindow = summaryMeta?.tail_window
    ?? storedSummary?.tail_window
    ?? summaryQuery.data?.market?.tail_window
    ?? false
  const eligibleCount = summaryMeta?.eligible_count
    ?? storedSummary?.eligible_count
    ?? summaryQuery.data?.market?.eligible_count
    ?? 0
  const marketWarnings = summaryMeta?.market_warnings
    ?? storedSummary?.market_warnings
    ?? summaryQuery.data?.market?.warnings
    ?? []

  return (
    <div className="min-h-full bg-base">
      <PageHeader
        title="快讯"
        titleExtra={<Newspaper className="h-4 w-4 text-accent" />}
        subtitle={formatSyncTime(syncStatus?.last_success_at)}
        right={
          <button
            type="button"
            onClick={() => refreshMutation.mutate()}
            disabled={Boolean(syncing)}
            className="inline-flex items-center gap-1.5 rounded-btn border border-border bg-elevated px-2.5 py-1.5 text-xs text-secondary transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
            title="同步财联社快讯"
          >
            <RefreshCw className={cn('h-3.5 w-3.5', syncing && 'animate-spin')} />
            {syncing ? '同步中' : '刷新'}
          </button>
        }
      />

      <main className="mx-auto w-full max-w-[1100px] px-3 py-3 sm:px-5 sm:py-4">
        {syncStatus && !syncStatus.backfill_completed && (
          <div className="mb-3 flex items-center gap-2 border border-accent/25 bg-accent/[0.06] px-3 py-2 text-xs text-accent">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            正在回补最近 7 天快讯，已有内容可正常浏览
          </div>
        )}

        {syncStatus?.last_error && (
          <div className="mb-3 flex items-start gap-2 border border-warning/30 bg-warning/[0.06] px-3 py-2 text-xs text-warning">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span className="min-w-0 break-words">最近同步失败：{syncStatus.last_error}，当前展示已保存内容</span>
          </div>
        )}

        <section className="mb-3 border border-border bg-surface" aria-label="今日新闻与盘面分析">
          <div className="flex flex-col gap-2 border-b border-border px-3 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:px-4">
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 text-sm font-semibold text-foreground">
                <Sparkles className="h-3.5 w-3.5 text-accent" />
                今日新闻与盘面分析
                {summaryQuery.data?.stale && (
                  <span className="rounded-sm border border-warning/30 bg-warning/10 px-1.5 py-0.5 text-[10px] font-normal text-warning">
                    有新快讯
                  </span>
                )}
                {!summaryQuery.isLoading && (
                  <span className={cn(
                    'rounded-sm border px-1.5 py-0.5 text-[10px] font-normal',
                    marketReady
                      ? 'border-accent/30 bg-accent/10 text-accent'
                      : 'border-warning/30 bg-warning/10 text-warning',
                  )}>
                    {marketReady ? `候选池 ${eligibleCount}` : '盘面待校验'}
                  </span>
                )}
                {tailWindow && (
                  <span className="rounded-sm border border-accent/30 bg-accent/10 px-1.5 py-0.5 text-[10px] font-normal text-accent">
                    尾盘窗口
                  </span>
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 text-[10px] text-muted">
                <span>{summaryCount} 条去重快讯</span>
                {summaryLatestAt && <span>截至 {formatPublishedAt(summaryLatestAt).time}</span>}
                {marketSnapshotAt && <span>盘面 {formatPublishedAt(marketSnapshotAt).time}</span>}
                {summaryGeneratedAt && <span>{formatSyncTime(summaryGeneratedAt).replace('最近同步', '生成于')}</span>}
              </div>
            </div>
            <button
              type="button"
              onClick={startSummary}
              disabled={summaryRunning || summaryQuery.isLoading}
              className="inline-flex h-8 shrink-0 items-center justify-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              title={storedSummary ? '重新分析今日新闻与盘面' : '分析今日新闻与盘面'}
            >
              {summaryRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              {summaryRunning ? '分析中' : storedSummary ? '重新分析' : '生成报告'}
            </button>
          </div>

          {!summaryRunning && marketWarnings.length > 0 && (
            <div className="flex items-start gap-2 border-b border-warning/20 bg-warning/[0.04] px-4 py-2 text-[11px] text-warning">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{marketWarnings[0]}</span>
            </div>
          )}

          {summaryRunning && !visibleSummary && (
            <div className="flex min-h-28 flex-col items-center justify-center gap-2 px-4 py-6 text-center">
              <Loader2 className="h-5 w-5 animate-spin text-accent" />
              <div className="text-xs text-secondary">{summaryProgress}</div>
            </div>
          )}
          {summaryError && (
            <div className="flex items-start gap-2 border-b border-danger/20 bg-danger/[0.05] px-4 py-2.5 text-xs text-danger">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{summaryError}</span>
            </div>
          )}
          {visibleSummary ? (
            <div className="max-h-[32rem] overflow-y-auto px-4 py-3 sm:px-5">
              <MarkdownRenderer content={visibleSummary} />
              {summaryPhase === 'streaming' && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-middle" />
              )}
            </div>
          ) : !summaryRunning && !summaryError ? (
            <div className="flex min-h-24 items-center justify-center px-4 py-5 text-xs text-muted">
              尚无今日分析报告
            </div>
          ) : null}
        </section>

        {newsQuery.isLoading ? (
          <div className="border border-border bg-surface">
            {Array.from({ length: 7 }).map((_, index) => (
              <div key={index} className="grid gap-3 border-b border-border/70 px-4 py-4 last:border-b-0 sm:grid-cols-[5.5rem_minmax(0,1fr)]">
                <div className="h-4 w-12 animate-pulse bg-elevated" />
                <div className="space-y-2">
                  <div className="h-4 w-2/3 animate-pulse bg-elevated" />
                  <div className="h-3 w-full animate-pulse bg-elevated" />
                </div>
              </div>
            ))}
          </div>
        ) : newsQuery.isError && items.length === 0 ? (
          <div className="flex min-h-64 flex-col items-center justify-center border border-border bg-surface px-6 text-center">
            <AlertTriangle className="h-6 w-6 text-warning" />
            <div className="mt-3 text-sm font-medium text-foreground">快讯加载失败</div>
            <button
              type="button"
              onClick={() => newsQuery.refetch()}
              className="mt-3 inline-flex items-center gap-1 text-xs text-accent hover:text-accent/80"
            >
              <RefreshCw className="h-3 w-3" />重试
            </button>
          </div>
        ) : items.length === 0 ? (
          <div className="flex min-h-64 flex-col items-center justify-center border border-border bg-surface px-6 text-center">
            <Clock3 className="h-6 w-6 text-muted" />
            <div className="mt-3 text-sm font-medium text-foreground">暂无快讯</div>
            <p className="mt-1 text-xs text-muted">后台正在获取财联社数据</p>
          </div>
        ) : (
          <>
            <section className="overflow-hidden border border-border bg-surface" aria-label="财联社快讯列表">
              {items.map(item => <NewsRow key={`${item.source}:${item.news_id}`} item={item} />)}
            </section>

            {newsQuery.hasNextPage && (
              <div className="flex justify-center py-4">
                <button
                  type="button"
                  onClick={() => newsQuery.fetchNextPage()}
                  disabled={newsQuery.isFetchingNextPage}
                  className="inline-flex min-w-28 items-center justify-center gap-1.5 rounded-btn border border-border bg-elevated px-3 py-1.5 text-xs text-secondary transition-colors hover:text-foreground disabled:opacity-50"
                >
                  {newsQuery.isFetchingNextPage && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  {newsQuery.isFetchingNextPage ? '加载中' : '加载更多'}
                </button>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
