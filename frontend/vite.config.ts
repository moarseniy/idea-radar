import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Сборка кладётся в app/static/app и отдаётся FastAPI; в dev /api проксируется на бэкенд.
export default defineConfig({
  plugins: [react()],
  base: '/static/app/',
  build: { outDir: '../app/static/app', emptyOutDir: true },
  server: { proxy: { '/api': process.env.API_URL ?? 'http://127.0.0.1:8091' } },
})
