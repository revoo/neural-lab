/** Minimal hash router: works on any static host with zero configuration. */
import { useSyncExternalStore } from 'react';

export type Route =
  | { page: 'home' }
  | { page: 'lesson'; id: string; anchor?: string }
  | { page: 'glossary'; term?: string }
  | { page: 'review' }
  | { page: 'timeline'; focus?: string }
  | { page: 'notfound' };

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  if (parts.length === 0) return { page: 'home' };
  if (parts[0] === 'learn' && parts[1]) return { page: 'lesson', id: parts[1], anchor: parts[2] };
  if (parts[0] === 'glossary') return { page: 'glossary', term: parts[1] };
  if (parts[0] === 'review') return { page: 'review' };
  if (parts[0] === 'timeline') return { page: 'timeline', focus: parts[1] };
  return { page: 'notfound' };
}

const subscribe = (cb: () => void) => {
  window.addEventListener('hashchange', cb);
  return () => window.removeEventListener('hashchange', cb);
};

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash, () => '');
  return parseHash(hash);
}

export const lessonHref = (id: string) => `#/learn/${id}`;
export const glossaryHref = (term?: string) => (term ? `#/glossary/${term}` : '#/glossary');
export const timelineHref = (focus?: string) => (focus ? `#/timeline/${focus}` : '#/timeline');
