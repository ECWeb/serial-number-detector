<template>
  <div class="flex min-h-screen bg-gray-800 text-gray-200">
    <Sidebar />
    <div class="flex-1 flex flex-col">
      <Navbar />
      <main class="flex-1 p-6 bg-[#1e1e2f] text-gray-200">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<script setup>
import Sidebar from '@/components/Sidebar.vue'
import Navbar from '@/components/Navbar.vue'
</script>
