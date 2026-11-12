// Entry point: registers the lazily loaded pages, sets the footer mode text and starts the router.
import { start, route, lazy, app, isStatic, $ } from './core.js';

const PAGES = { 'home.js': ['home'], 'simulate.js': ['simulate'], 'benchmark.js': ['benchmark'], 'method.js': ['method'],
  'model.js': ['model'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the simulator runs in your browser (JS port of the Python code); benchmark computed in Python'
  : 'local app: Python server on this machine';
start();
