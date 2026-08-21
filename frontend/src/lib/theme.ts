/** Theme selection, persisted across sessions. */

export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'checkbox-detection.theme'

/**
 * The theme to start in.
 *
 * A stored choice wins; otherwise the OS preference decides. Defaulting to light rather than
 * to the OS would override a deliberate system-wide setting on first visit, which is exactly
 * the thing people who run dark systems find rude.
 *
 * Wrapped in try/catch because `localStorage` throws outright in a sandboxed iframe and in
 * Safari's private mode — a theme preference is not worth taking the whole app down for.
 */
export function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // Storage unavailable; fall through to the system preference.
  }
  if (typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

/**
 * Apply a theme to the document and remember it.
 *
 * The attribute goes on `<html>`, not on the app's root element, so that fixed-position
 * surfaces — the lightbox in particular — inherit the same tokens as everything else.
 */
export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme)
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    // Persistence is a nicety; the theme still applies for this session.
  }
}
