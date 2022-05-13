import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    mode: 'hash',
    name: 'Home',
    component: () => import('../components/FlowMapGAT.vue'),
    meta: {
      title: 'Index'
    }
  },
  {
    path: '/person/:name/loc/:address/start/:sm/end/:em',
    mode: 'hash',
    component: () => import('../components/ValidSource.vue'),
    meta: {
      title: 'ValidSource'
    }
  },
]

const router = createRouter({
  history: createWebHistory(),
  scrollBehavior: () => ({ y: 0 }),
  routes: routes
})

export default router