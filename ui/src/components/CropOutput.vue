<!-- components/CropOutput.vue -->
<template>
  <div v-if="croppedImage" class="">
    <h3 class="font-bold">Output:</h3>
    <img :src="croppedImage" class="border mt-2 max-w-full" style="width: 100%; max-width: 300px;" />
  </div>
</template>

<script setup>
defineProps({
  croppedImage: String
})
</script>
