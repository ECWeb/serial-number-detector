<template>
  <div>
    <input type="file" accept="image/*" @change="onFileChange" class="mb-2" />

    <div v-if="imageUrl" class="relative inline-block border p-2" style="width: 100%; max-width: 600px; height: 400px;">
      <img
        ref="imageElement"
        :src="imageUrl"
        class="max-w-full max-h-[300px]"
        :style="rotationStyle"
      />
    </div>

    <div v-if="cropper" class="mt-4 flex flex-wrap gap-2">
      <button @click="rotateLeft" class="px-4 py-1 bg-blue-600 text-white rounded">Rotate Left</button>
      <button @click="rotateRight" class="px-4 py-1 bg-blue-600 text-white rounded">Rotate Right</button>
      <button @click="cropImage" class="px-4 py-1 bg-green-600 text-white rounded">Crop</button>
      <button @click="resetImage" class="px-4 py-1 bg-red-600 text-white rounded">Reset</button>
    </div>

    <div v-if="croppedImage" class="mt-4">
      <h3 class="font-bold">Cropped Image:</h3>
      <img :src="croppedImage" class="border mt-2 max-w-full" style="width: 100%; max-width: 300px;"/>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import Cropper from 'cropperjs'
import 'cropperjs/dist/cropper.css'

const imageUrl = ref(null)
const croppedImage = ref(null)
const cropper = ref(null)
const imageElement = ref(null)
const rotation = ref(0)

// computed style for image rotation
const rotationStyle = computed(() => {
  return {
    transform: `rotate(${rotation.value}deg)`,
    transition: 'transform 0.3s ease',
  }
})

// Watch for image load and initialize cropper
watch(imageUrl, () => {
  nextTick(() => {
    if (imageElement.value) {
      if (cropper.value) {
        cropper.value.destroy()
      }
      cropper.value = new Cropper(imageElement.value, {
        viewMode: 1,
        autoCropArea: 1,
        responsive: true,
      })
    }
  })
})

function onFileChange(e) {
  const file = e.target.files[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => {
    imageUrl.value = reader.result
    croppedImage.value = null
    rotation.value = 0
  }
  reader.readAsDataURL(file)
}

function cropImage() {
  if (!cropper.value) return
  const canvas = cropper.value.getCroppedCanvas()
  croppedImage.value = canvas.toDataURL('image/png')
}

// function resetImage() {
//   try {
//     if (cropper.value) {
//       cropper.value.clear()    // clears the crop box
//       cropper.value.crop()     // reactivates default crop box
//     }
//   } catch (error) {
//     console.error('Error resetting cropper:', error)
//   }
// }

function resetImage() {
  if (cropper.value) {
    cropper.value.reset() // Resets the cropper to its initial state
    croppedImage.value = null // Clear the cropped image
    rotation.value = 0 // Reset rotation
    // imageUrl.value = null // Clear the image URL
    // imageElement.value.src = '' // Clear the image element source  
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
// function rotateLeft() {
//   rotation.value = (rotation.value - 90) % 360
//   cropper.value.rotateTo(rotation.value)
// }

// function rotateRight() {
//   rotation.value = (rotation.value + 90) % 360
//   cropper.value.rotateTo(rotation.value)
// }

</script>
