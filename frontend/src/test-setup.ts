/**
 * jsdom implements neither object URLs nor layout, so two browser APIs this UI relies on are
 * missing under test. Stubbing them here rather than guarding the component keeps the
 * production code free of test-only branches.
 */
import '@testing-library/jest-dom/vitest'

let counter = 0

// Used to display the chosen image without uploading it anywhere.
if (!URL.createObjectURL) {
  URL.createObjectURL = () => `blob:test/${++counter}`
}
if (!URL.revokeObjectURL) {
  URL.revokeObjectURL = () => undefined
}
