import { useEffect, useState } from 'react';
import { MDXProvider } from '@mdx-js/react';
import { useRoute } from './router';
import { Sidebar } from './pages/Sidebar';
import { Home } from './pages/Home';
import { LessonPage } from './pages/LessonPage';
import { GlossaryPage } from './pages/GlossaryPage';
import { ReviewPage } from './pages/ReviewPage';
import { findLesson } from './curriculum';
import { mdxComponents } from './mdxComponents';

function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const t = document.documentElement.dataset.theme;
    if (t === 'light' || t === 'dark') return t;
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('nl-theme', next);
    } catch {
      /* ignore */
    }
    setTheme(next);
  };
  return { theme, toggle };
}

export function App() {
  const route = useRoute();
  const { theme, toggle } = useTheme();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
    window.scrollTo(0, 0);
  }, [route.page, route.page === 'lesson' ? route.id : '']);

  const lesson = route.page === 'lesson' ? findLesson(route.id) : undefined;
  const crumbs =
    route.page === 'lesson' && lesson
      ? `${lesson.module.number}. ${lesson.module.title}  ›  ${lesson.title}`
      : route.page === 'glossary'
        ? 'Glossary'
        : route.page === 'review'
          ? 'Review'
          : 'Neural Lab';

  useEffect(() => {
    document.title = lesson ? `${lesson.title} · Neural Lab` : 'Neural Lab';
  }, [lesson]);

  return (
    <MDXProvider components={mdxComponents}>
      <div className="app">
        <Sidebar route={route} open={navOpen} />
        {navOpen && <div className="scrim" onClick={() => setNavOpen(false)} />}
        <div className="main">
          <header className="topbar">
            <button className="icon-btn menu-btn" onClick={() => setNavOpen(true)} aria-label="Open navigation">☰</button>
            <div className="crumbs">{crumbs}</div>
            <a className="icon-btn" href="#/glossary">Glossary</a>
            <button className="icon-btn" onClick={toggle} aria-label="Toggle colour theme">
              {theme === 'dark' ? '☀ Light' : '☾ Dark'}
            </button>
          </header>
          {route.page === 'home' && <Home />}
          {route.page === 'lesson' && (lesson ? <LessonPage key={lesson.id} lesson={lesson} /> : <NotFound />)}
          {route.page === 'glossary' && <GlossaryPage term={route.term} />}
          {route.page === 'review' && <ReviewPage />}
          {route.page === 'notfound' && <NotFound />}
        </div>
      </div>
    </MDXProvider>
  );
}

function NotFound() {
  return (
    <div className="page">
      <h1>Nothing here (yet)</h1>
      <p>
        That page doesn’t exist. <a href="#/">Back to the curriculum →</a>
      </p>
    </div>
  );
}
