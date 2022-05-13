<template>
  <el-container>
    <el-main>
      <router-view />
    </el-main>
  </el-container>
</template>

<script>
import FlowMapGAT from './components/FlowMapGAT.vue'
import ValidSource from './components/ValidSource.vue'

export default {
  name: 'App',
  components: {
    FlowMapGAT,
    ValidSource
  }
}
</script>
