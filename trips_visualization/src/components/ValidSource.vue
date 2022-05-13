<template>
  <h1 style="margin-left:3%"> The news source of <i>{{ $route.params.name }}</i> 's trip in <i>{{ $route.params.address }}</i></h1>

  <p style="margin-left:3%">
    From {{ $route.params.sm }} to {{ $route.params.em }}
  </p>

  <div id="text-list">
      <el-table :data="tableData" style="width: 100%">
      <el-table-column type="expand">
          <template #default="props">
            <p v-for="(text, index) in props.row.text_list">
              {{index+1}}. {{ text }}
              <a :href="props.row.url_list[index]">source</a>
            </p>
          </template>
        </el-table-column>
        <el-table-column label="Date" prop="date" />
        <el-table-column label="Wiki Label" prop="wiki_label" />
      </el-table>
  </div>

</template>

<script>
import { defineComponent } from 'vue';
import axios from 'axios';

export default defineComponent({
    data() {
      var tableData = [
        {
          "date": "Loading...",
          "wiki_label": "Loading...",
        }
      ];
      return {
        tableData
      };
    },
    mounted(){
      var thisPage = this;
      axios.get("http://124.223.93.31:20512/api/get_itin_info",{params:{
        name: thisPage.$route.params.name, loc: thisPage.$route.params.address,
        time_start: thisPage.$route.params.sm, time_end: thisPage.$route.params.em}})
          .then((res) => {
        this.tableData = res['data']['info_list'];
      });
    },
})
</script>

<style scoped>
#text-list p{
  font-size: 20px;
  width: 80%;
  margin-left: 3%;
}

#text-list div{
  font-size: 23px;
}

#text-list a{
  font-size: 23px;
}

#text-list a:link{
  text-decoration:none;
  color:#2a38b5;
}  

#text-list a:visited{
  text-decoration:none;
  color:#2a38b5; 
}

#text-list a:hover{
  text-decoration:underline;
  color:#3a70ae;
}

#text-list a:active{
  text-decoration:underline;
  color:#e71f1f;
}

</style>