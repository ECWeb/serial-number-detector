import { createRouter, createWebHistory } from 'vue-router'
import DefaultLayout from '@/layouts/DefaultLayout.vue'
import Home from '@/views/HomeView.vue'
import Cropper from '@/views/CropperPage.vue'
import About from '@/views/AboutView.vue'

const routes = [
  {
    path: '/',
    component: DefaultLayout,
    children: [
      { path: '', component: Cropper },
      // { path: 'cropper', component: Cropper },
      // { path: 'about', component: About },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
