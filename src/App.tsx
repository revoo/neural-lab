import { useCallback, useEffect, useState } from 'react';
import { MDXProvider } from '@mdx-js/react';
import { useRoute } from './router';
import { Sidebar } from './pages/Sidebar';
import { Home } from './pages/Home';
import { LessonPage } from './pages/LessonPage';
import { GlossaryPage } from './pages/GlossaryPage';
import { ReviewPage } from './pages/ReviewPage';
import { TimelinePage } from './pages/TimelinePage';
import { findLesson } from './curriculum';
import { mdxComponents } from './mdxComponents';
import { prefersReducedMotion, useRain, useTheme, type Theme } from './theme';
import { DigitalRain } from './components/DigitalRain';
import { BootSequence, ThemeSwitch } from './components/ThemeSwitch';

export function App() {
  const route = useRoute();
  const { theme, setTheme } = useTheme();
  const { rain, setRain } = useRain();
  const [booting, setBooting] = useState(false);
  const endBoot = useCallback(() => setBooting(false), []);
  const changeTheme = (t: Theme) => {
    if (t === 'matrix' && theme !== 'matrix' && !prefersReducedMotion()) setBooting(true);
    setTheme(t);
  };
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
          : route.page === 'timeline'
            ? 'Timeline'
            : 'Neural Lab';

  useEffect(() => {
    document.title = lesson ? `${lesson.title} · Neural Lab` : 'Neural Lab';
  }, [lesson]);

  return (
    <MDXProvider components={mdxComponents}>
      {theme === 'matrix' && <DigitalRain animate={rain} />}
      {booting && <BootSequence onDone={endBoot} />}
      <div className="app">
        <Sidebar route={route} open={navOpen} />
        {navOpen && <div className="scrim" onClick={() => setNavOpen(false)} />}
        <div className="main">
          <header className="topbar">
            <button className="icon-btn menu-btn" onClick={() => setNavOpen(true)} aria-label="Open navigation">☰</button>
            <div className="crumbs">{crumbs}</div>
            <a className="icon-btn" href="#/glossary">Glossary</a>
            {theme === 'matrix' && (
              <button
                className="icon-btn rain-btn"
                onClick={() => setRain(!rain)}
                aria-pressed={rain}
                aria-label="Digital rain"
                title={rain ? 'Pause the digital rain' : 'Resume the digital rain'}
              >
                <span aria-hidden="true">{rain ? '❚❚' : '▶'}</span>
                <span className="rain-label">Rain</span>
              </button>
            )}
            <ThemeSwitch theme={theme} onChange={changeTheme} />
          </header>
          {route.page === 'home' && <Home />}
          {route.page === 'lesson' && (lesson ? <LessonPage key={lesson.id} lesson={lesson} /> : <NotFound />)}
          {route.page === 'glossary' && <GlossaryPage term={route.term} />}
          {route.page === 'review' && <ReviewPage />}
          {route.page === 'timeline' && <TimelinePage focus={route.focus} />}
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
