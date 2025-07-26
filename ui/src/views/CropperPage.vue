<!-- pages/CropperPage.vue -->
<template>
  <div class="flex flex-col md:flex-row gap-6">
    <CropInput @cropped="handleCropped" class="w-full rounded-md bg-gray-800 p-3 md:w-1/2" />
    <CropOutput :croppedImage="croppedImage" class="w-full rounded-md bg-gray-800 p-3 md:w-1/2" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import CropInput from '@/components/ImageCropper.vue'
import CropOutput from '@/components/CropOutput.vue'

const croppedImage = ref(null)

function handleCropped(dataUrl) {
  croppedImage.value = dataUrl
}
</script>
