import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  ArrowUpRight,
  ExternalLink,
  Loader2,
  Newspaper,
  Volume2,
  VolumeX,
} from 'lucide-react'
import { Link } from 'react-router-dom'

import { api, type FinanceNewsItem } from '@/lib/api'
import { cn } from '@/lib/cn'
import { financeNewsTitle, isImportantFinanceNews } from '@/lib/financeNews'
import { QK } from '@/lib/queryKeys'
import { storage } from '@/lib/storage'
import {
  activateVoice,
  isVoiceSupported,
  speakVoiceText,
  stopVoice,
} from '@/lib/voiceBroadcast'

const EMPTY_ITEMS: FinanceNewsItem[] = []
const MAX_STORED_IDS = 200
const MAX_SPEAK_ITEMS = 3
const DISPLAY_ITEMS = 12

function itemKey(item: FinanceNewsItem) {
  return `${item.source}:${item.news_id}`
}

function rememberItems(items: FinanceNewsItem[]) {
  if (items.length === 0) return
  const current = storage.dashboardNewsSpokenIds.get([])
  const merged = [...new Set([...items.map(itemKey), ...current])]
  storage.dashboardNewsSpokenIds.set(merged.slice(0, MAX_STORED_IDS))
}

function formatNewsTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--:--'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Shanghai',
  }).format(date)
}

function NewsItem({ item }: { item: FinanceNewsItem }) {
  const important = isImportantFinanceNews(item)
  const subject = item.subjects.find(value => value.subject_name)?.subject_name
  const stocks = item.stocks.filter(value => value.stock_name || value.stock_code)
  const title = financeNewsTitle(item)

  return (
    <article
      className={cn(
        'relative border-b border-border/60 px-2 py-2 last:border-b-0',
        important && 'bg-danger/[0.045]',
      )}
    >
      {important && <span className="absolute inset-y-0 left-0 w-0.5 bg-danger" aria-hidden />}
      <a
        href={item.url}
        target="_blank"
        rel="noreferrer"
        className="group flex min-w-0 items-start gap-2"
        aria-label={`打开财联社新闻：${title}`}
      >
        <time className="w-[4.8rem] shrink-0 pt-0.5 font-mono text-[9px] text-muted">
          {formatNewsTime(item.published_at)}
        </time>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-start gap-1.5">
            {important && (
              <span className="mt-0.5 shrink-0 rounded-sm border border-danger/35 bg-danger/10 px-1 py-px text-[8px] font-semibold text-danger">
                重点
              </span>
            )}
            <h3
              className={cn(
                'line-clamp-2 min-w-0 flex-1 text-[11px] font-medium leading-[1.45] text-secondary transition-colors group-hover:text-accent',
                important && 'text-foreground',
              )}
              title={title}
            >
              {title}
            </h3>
            <ExternalLink className="mt-0.5 h-3 w-3 shrink-0 text-muted/50 transition-colors group-hover:text-accent" />
          </div>
          {(subject || stocks.length > 0) && (
            <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1 text-[9px] text-muted">
              {subject && <span className="max-w-full truncate">{subject}</span>}
              {stocks.slice(0, 2).map(stock => (
                <span
                  key={`${stock.stock_code}:${stock.stock_name}`}
                  className="inline-flex max-w-full items-center gap-1 rounded-sm bg-accent/10 px-1 py-px text-accent"
                >
                  <span className="truncate">{stock.stock_name || stock.stock_code}</span>
                  {stock.stock_name && stock.stock_code && (
                    <span className="shrink-0 font-mono text-[8px] text-accent/70">{stock.stock_code}</span>
                  )}
                </span>
              ))}
              {stocks.length > 2 && <span className="text-accent/70">+{stocks.length - 2}</span>}
            </div>
          )}
        </div>
      </a>
    </article>
  )
}

export function DashboardNewsPanel() {
  const [voiceEnabled, setVoiceEnabled] = useState(
    () => storage.dashboardNewsVoice.get(false),
  )
  const baselineReady = useRef(false)
  const voiceSupported = isVoiceSupported()
  const news = useQuery({
    queryKey: QK.financeNewsDashboard,
    queryFn: () => api.financeNewsList(50),
    refetchInterval: () => document.visibilityState === 'visible' ? 60_000 : false,
    placeholderData: previous => previous,
  })
  const items = news.data?.items ?? EMPTY_ITEMS
  const importantItems = items.filter(isImportantFinanceNews)
  const latestItems = items.slice(0, DISPLAY_ITEMS)
  const latestDate = items[0]?.published_at.slice(0, 10)
  const pinnedImportant = items.find(item =>
    isImportantFinanceNews(item)
    && item.published_at.slice(0, 10) === latestDate
    && !latestItems.some(latest => itemKey(latest) === itemKey(item)))
  const pinnedStock = items.find(item =>
    item.stocks.length > 0
    && item.published_at.slice(0, 10) === latestDate
    && !latestItems.some(latest => itemKey(latest) === itemKey(item))
    && itemKey(item) !== (pinnedImportant ? itemKey(pinnedImportant) : ''))
  const pinnedItems = [pinnedImportant, pinnedStock].filter(
    (item): item is FinanceNewsItem => Boolean(item),
  )
  const visibleItems = [
    ...pinnedItems,
    ...latestItems.filter(item =>
      !pinnedItems.some(pinned => itemKey(pinned) === itemKey(item))),
  ].slice(0, DISPLAY_ITEMS)

  useEffect(() => {
    if (news.isLoading) return

    if (!baselineReady.current) {
      rememberItems(importantItems)
      baselineReady.current = true
      return
    }
    if (!voiceEnabled || importantItems.length === 0) return

    const spoken = new Set(storage.dashboardNewsSpokenIds.get([]))
    const fresh = importantItems.filter(item => !spoken.has(itemKey(item)))
    if (fresh.length === 0) return

    const spokenNow = fresh.slice(0, MAX_SPEAK_ITEMS).reverse()
    const text = `重点快讯。${spokenNow.map(financeNewsTitle).join('。')}`
    speakVoiceText(text)
    rememberItems(fresh)
  }, [importantItems, news.isLoading, voiceEnabled])

  const toggleVoice = () => {
    if (!voiceSupported) return
    const next = !voiceEnabled
    if (next) {
      activateVoice()
      rememberItems(importantItems)
    } else {
      stopVoice()
    }
    storage.dashboardNewsVoice.set(next)
    setVoiceEnabled(next)
  }

  return (
    <section
      className="overflow-hidden rounded-card border border-border bg-surface/80 shadow-[0_1px_2px_hsl(var(--border)/0.4)] backdrop-blur-sm transition-shadow hover:shadow-[0_2px_8px_hsl(var(--border)/0.5)]"
      aria-label="最新快讯"
    >
      <div className="flex h-8 items-center justify-between gap-2 border-b border-border/70 px-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <Newspaper className="h-3.5 w-3.5 shrink-0 text-accent" />
          <h2 className="shrink-0 text-xs font-semibold text-foreground">最新快讯</h2>
          {news.data?.sync_status.syncing ? (
            <Loader2 className="h-3 w-3 animate-spin text-accent" aria-label="快讯同步中" />
          ) : (
            <span className="inline-flex items-center gap-1 truncate text-[9px] text-muted">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              60秒更新
            </span>
          )}
          {(news.isError || news.data?.sync_status.last_error) && (
            <AlertTriangle
              className="h-3 w-3 shrink-0 text-warning"
              aria-label="快讯同步异常，正在展示已保存内容"
            />
          )}
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <button
            type="button"
            onClick={toggleVoice}
            disabled={!voiceSupported}
            className={cn(
              'inline-flex h-6 w-6 items-center justify-center rounded text-muted transition-colors hover:bg-accent/10 hover:text-accent disabled:cursor-not-allowed disabled:opacity-35',
              voiceEnabled && 'bg-accent/10 text-accent',
            )}
            aria-label={voiceEnabled ? '关闭重点快讯播报' : '开启重点快讯播报'}
            title={
              voiceSupported
                ? `${voiceEnabled ? '关闭' : '开启'}新到重点快讯播报`
                : '当前浏览器不支持语音播报'
            }
          >
            {voiceEnabled
              ? <Volume2 className="h-3.5 w-3.5" />
              : <VolumeX className="h-3.5 w-3.5" />}
          </button>
          <Link
            to="/news"
            className="inline-flex h-6 w-6 items-center justify-center rounded text-muted transition-colors hover:bg-accent/10 hover:text-accent"
            title="查看全部快讯"
            aria-label="查看全部快讯"
          >
            <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>

      {news.isLoading ? (
        <div className="space-y-2 p-2" aria-label="快讯加载中">
          {Array.from({ length: 5 }).map((_, index) => (
            <div key={index} className="flex gap-2">
              <div className="h-3 w-[4.8rem] animate-pulse bg-elevated" />
              <div className="h-3 flex-1 animate-pulse bg-elevated" />
            </div>
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="px-3 py-7 text-center text-[11px] text-muted">
          {news.isError ? '快讯加载失败' : '暂无快讯，后台正在同步'}
        </div>
      ) : (
        <div className="max-h-[23rem] overflow-y-auto overscroll-contain" aria-live="polite">
          {visibleItems.map(item => <NewsItem key={itemKey(item)} item={item} />)}
        </div>
      )}
    </section>
  )
}
