<template>

  <div>
    <!-- 📥 Drag-and-Drop Zone -->
    <div
      class="border-2 border-dashed border-gray-400 p-6 rounded text-center cursor-pointer transition hover:border-blue-500"
      @dragover.prevent
      @dragenter.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="handleDrop"
      @click="triggerFileInput"
      :class="{ 'bg-blue-50': isDragging }"
    >
      <p class="text-gray-600">
        Drag & drop an image here, or <span class="underline text-blue-600">click to select</span>
      </p>
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        class="hidden"
        @change="onFileChange"
      />
    </div>

    <!-- 🖼️ Preview + Cropper -->
    <div v-if="imageUrl" class="mt-4 border p-2" style="width: 100%; max-height: 400px;">
      <img
        ref="imageElement"
        :src="imageUrl"
        class="max-h-[300px]"
        :style="rotationStyle"
      />
    </div>

    <!-- 🧰 Controls -->
    <div v-if="cropper" class="mt-4 flex flex-wrap gap-2">
      <button @click="rotateLeft" class="px-4 py-1 bg-blue-600 text-white rounded">Rotate Left</button>
      <button @click="rotateRight" class="px-4 py-1 bg-blue-600 text-white rounded">Rotate Right</button>
      <button @click="cropImage" class="px-4 py-1 bg-green-600 text-white rounded">Crop</button>
      <button @click="resetImage" class="px-4 py-1 bg-red-600 text-white rounded">Reset</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import Cropper from 'cropperjs'
import 'cropperjs/dist/cropper.css'

const emit = defineEmits(['cropped'])

const imageUrl = ref(null)
const cropper = ref(null)
const imageElement = ref(null)
const fileInput = ref(null)
const rotation = ref(0)
const isDragging = ref(false)

const rotationStyle = computed(() => ({
  transform: `rotate(${rotation.value}deg)`,
  transition: 'transform 0.3s ease',
}))

watch(imageUrl, () => {
  nextTick(() => {
    if (imageElement.value) {
      if (cropper.value) cropper.value.destroy()
      cropper.value = new Cropper(imageElement.value, {
        viewMode: 1,
        autoCropArea: 1,
        responsive: true,
      })
    }
  })
})

function triggerFileInput() {
  fileInput.value?.click()
}

function onFileChange(e) {
  const file = e.target.files[0]
  if (file) loadImage(file)
}

function handleDrop(e) {
  isDragging.value = false
  const file = e.dataTransfer.files[0]
  if (file) loadImage(file)
}

function loadImage(file) {
  const reader = new FileReader()
  reader.onload = () => {
    imageUrl.value = reader.result
    rotation.value = 0
    emit('cropped', null)
  }
  reader.readAsDataURL(file)
}

function cropImage() {
  if (!cropper.value) return
  const canvas = cropper.value.getCroppedCanvas()
  const croppedData = canvas.toDataURL('image/png')
  emit('cropped', croppedData)
}

function resetImage() {
  if (cropper.value) {
    cropper.value.reset()
    rotation.value = 0
    emit('cropped', null)
  }
}

function rotateLeft() {
  rotation.value -= 90
  cropper.value.rotate(-1.5)
}

function rotateRight() {
  rotation.value += 90
  cropper.value.rotate(1.5)
}
</script>
<style>
    .cropper-container {
      width: 100% !important;
      max-width: 100%;
    }
  </style>
