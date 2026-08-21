import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// The dev server binds 0.0.0.0 so the container-published port is reachable from the host;
// binding localhost inside a container makes the app appear dead from outside it.
export default defineConfig({
  plugins: [react()],
  server: { host: '0.0.0.0', port: 5173 },
  preview: { host: '0.0.0.0', port: 5173 },
  test: {
    environment: 'jsdom',
    globals: true,
    coverage: { provider: 'v8', reporter: ['text', 'lcov'], include: ['src/**/*.{ts,tsx}'] },
  },
})
