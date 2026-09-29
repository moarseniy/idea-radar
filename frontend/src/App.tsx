import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Empty } from './components/parts'
import { Layout } from './components/Layout'
import { Home } from './pages/Home'
import { CatalogPage, MethodPage, RegistryPage, StatsPage } from './pages/Other'
import { RunPage } from './pages/RunPage'
import { SignalPage } from './pages/SignalPage'
import { AppProvider } from './state'

export function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="runs/:runId" element={<RunPage />} />
            <Route path="signals/:signalId" element={<SignalPage />} />
            <Route path="stats" element={<StatsPage />} />
            <Route path="registry" element={<RegistryPage />} />
            <Route path="catalog" element={<CatalogPage />} />
            <Route path="method" element={<MethodPage />} />
            <Route path="*" element={<Empty title="Страница не найдена" text="Вернитесь к обзору сигналов." />} />
          </Route>
        </Routes>
      </AppProvider>
    </BrowserRouter>
  )
}
