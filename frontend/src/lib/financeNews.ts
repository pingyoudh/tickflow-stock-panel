import type { FinanceNewsItem } from './api'

export function financeNewsTitle(item: Pick<FinanceNewsItem, 'title' | 'content'>): string {
  const title = item.title.trim()
  if (title) return title

  const compact = item.content.replace(/\s+/g, ' ').trim()
  if (!compact) return '财联社快讯'
  const first = compact.split(/[。！？!?；;\n]/, 1)[0] || compact
  return first.length > 72 ? `${first.slice(0, 72)}…` : first
}

export function isImportantFinanceNews(
  item: Pick<FinanceNewsItem, 'level' | 'recommend'>,
): boolean {
  return item.recommend || item.level.trim().toUpperCase().startsWith('A')
}
