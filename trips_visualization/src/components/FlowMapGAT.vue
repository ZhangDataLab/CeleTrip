<template>
  <div class="demo-date-picker">
    <div class="block">
      <span class="demonstration">Choose a <i>politician</i> and <i>period of time</i></span>

      <el-row>
        <el-col :span="12">
          <el-select v-model="person_name" class="m-2" placeholder="Select Politician" @change="handleSelect">
            <el-option
              v-for="item in person_options"
              :key="item.value"
              :value="item.value"
            >
            </el-option>
          </el-select>
        </el-col>

        <el-col :span="12">
          <el-date-picker
            v-model="month_range"
            type="monthrange"
            range-separator="To"
            start-placeholder="Start month"
            end-placeholder="End month"
            :unlink-panels=false
            :disabled="disableCalendar"
            :disabled-date="ifDisabledDate"
            @change="handleCalendar"
          >
          </el-date-picker>
        </el-col>

        <el-col :span="12">
          <el-button
            type="primary"
            plain
            :disabled="disableButton"
            @click="queryItin">
            Query
          </el-button>
        </el-col>
      </el-row>

    </div>
  </div>

  <div class="map-wrapper">
    <div id="echartsMap"></div>
  </div>
</template>

<script>
import * as echarts from 'echarts';
import { RegisterWorldMap } from '../js/world.js';
import axios from 'axios';

import { defineComponent, ref, onMounted } from 'vue'

export default {
    name: 'FlowMapGAT',
    data() {
      return{
        disableCalendar: true,
        disableButton: true
      }
    },

    mounted(){
      var FlowMap = this;
      axios.get("http://xxx.xxx.xxx.xxx/api/get_gat_person_itin_by_time",{params:{name:'Vladimir Putin', time_start:'2019-07', time_end:'2022-01'}})
          .then((res) => {
        let capital = res['data']['capital'];
        let process_res = FlowMap.processItins(capital, res['data']['itins_pm']);
        let itinFrames = process_res[0]; let month_list = process_res[1];
        FlowMap.drawMap('Vladimir Putin', capital, res['data']['geocoord'], itinFrames, month_list);
      });
    },
    methods:{

      queryItin() {
        let s_year = this.month_range[0].getFullYear();
        let s_month = (this.month_range[0].getMonth() + 1).toString();
        if  (s_month.length == 1) {
          s_month =  "0"  + s_month; 
        };
        let time_start = s_year + "-" + s_month;

        let e_year = this.month_range[1].getFullYear();
        let e_month = (this.month_range[1].getMonth() + 1).toString();
        if  (e_month.length == 1) {
          e_month =  "0"  + e_month; 
        };
        let time_end = e_year + "-" + e_month;

        let FlowMap = this;
        let person_name = this.person_name;
        axios.get("http://xxx.xxx.xxx.xxx/api/get_gat_person_itin_by_time",{params:{name:person_name, time_start:time_start, time_end:time_end}})
          .then((res) => {
          let capital = res['data']['capital'];
          let process_res = FlowMap.processItins(capital, res['data']['itins_pm']);
          let itinFrames = process_res[0]; let month_list = process_res[1];
          FlowMap.drawMap(person_name, capital, res['data']['geocoord'], itinFrames, month_list);
        });
      },

      handleSelect(item) {
        this.disableCalendar = false;
      },

      handleCalendar(item) {
        this.disableButton = false;
      },

      drawMap(person_name, capital, geoCoordMap, itinFrames, month_list){
        document.getElementById('echartsMap').removeAttribute('_echarts_instance_');
        let chart = echarts.init(document.getElementById('echartsMap'));

        var planePath = 'path://M1705.06,1318.313v-89.254l-319.9-221.799l0.073-208.063c0.521-84.662-26.629-121.796-63.961-121.491c-37.332-0.305-64.482,36.829-63.961,121.491l0.073,208.063l-319.9,221.799v89.254l330.343-157.288l12.238,241.308l-134.449,92.931l0.531,42.034l175.125-42.917l175.125,42.917l0.531-42.034l-134.449-92.931l12.238-241.308L1705.06,1318.313z';
        
        var convertData = function(data) {
            var res = [];
            for (var i = 0; i < data.length; i++) {
                var dataItem = data[i];
                var fromCoord = geoCoordMap[dataItem[0].name];
                var toCoord = geoCoordMap[dataItem[1].name];
                if (fromCoord && toCoord) {
                    res.push([{
                        coord: fromCoord
                    }, {
                        coord: toCoord
                    }])
                }
            }
            return res;
        }

        var color = ['#9ae5fc', '#dcbf71'];

        var geoCoordMap = geoCoordMap;

        var series = new Array(); var options = new Array();
        for (let i = 0; i < itinFrames.length; i++){
          series.push([]);
        }

        itinFrames.forEach(function(itinFrame, i) {
          series[i].push({
            type: 'lines',
            zlevel: 1,
            effect: {
              show: true,
              period: 10,
              trailLength: 0.7,
              color: '#0ff',
              symbolSize: 3
            },
            lineStyle: {
              normal: {
              color: color[0],
              width: 0,
              curveness: -0.2
              }
            },
              data: convertData(itinFrame)
          }, {
            type: 'lines',
            zlevel: 2,
            symbolSize: 10,
            effect: {
              show: true,
              period: 10,
              trailLength: 0,
              symbol: planePath,
              symbolSize: 15
            },
            lineStyle: {
              normal: {
                color: color[0],
                width: 1,
                opacity: 0.6,
                curveness: -0.2
              }
            },
            data: convertData(itinFrame)
          }, {
            type: 'effectScatter',
            coordinateSystem: 'geo',
            zlevel: 3,
            rippleEffect: {
              brushType: 'fill'
            },
            label: {
              normal: {
                show: false,
                position: 'left',
                formatter: '{b}'
              }
            },
            itemStyle: {
              normal: {
                color: color[0]
              }
            },
            data: itinFrame.map(function(dataItem) {
              return {
                name: dataItem[1].name,
                value: geoCoordMap[dataItem[1].name],
                date_list: dataItem[1].date_list,
                symbolSize: 8 + 10*Math.log(dataItem[1].value),
              };
            })
          });
      });

        itinFrames.forEach(function(itinFrame, i) {
          series[i].push({
            type: 'effectScatter',
            coordinateSystem: 'geo',
            zlevel: 3,
            rippleEffect: {
                brushType: 'stroke'
            },
            label: {
                normal: {
                    show: true,
                    position: 'left',
                    formatter: '{b}'
                }
            },
            symbolSize: function(val) {
                return 15;
            },
            itemStyle: {
                normal: {
                    color: "#f00"
                }
            },
            data: [{
                name: capital,
                value: geoCoordMap[capital],
                label: {
                  normal: {
                    position: 'right'
                  }
                }
            }]
          });
        });

        itinFrames.forEach(function(itinFrame, i) {
          options.push({
            title: {
              text: month_list[i],
              subtext: 'beta version',
              textStyle: {
                color: '#fff',
                fontSize: 20
              },
              top: '10px',
              left: '10px'
            },
            tooltip: {
              trigger: 'item',
              formatter: (dataItem) => {
                if (dataItem.componentSubType != "effectScatter"){
                  return '';
                }
                if (typeof(dataItem.data.date_list) == "undefined"){
                  return dataItem.marker + dataItem.data.name;
                } else {
                  let text = dataItem.marker + dataItem.data.name;
                  for (let date in dataItem.data.date_list) {
                    text += '<br>' + dataItem.data.date_list[date];
                  }
                  return text;
                }
              },
              borderWidth: 2,
              backgroundColor: 'rgba(255, 255, 255, 0.8)'
            },
            series: series[i]
          });
        });

        chart.setOption({
            // baseOption
            timeline: {
                data: month_list
            },
            backgroundColor: '#101750',
            geo: {
                map: 'world',
                roam: true,
                itemStyle: {
                    normal: {
                        areaColor: '#7d7d7d'
                    },
                    emphasis: {
                        areaColor: '#2a333d'
                    }
                }
            },
            options: options
        });

        chart.on('dblclick', function(params) {
          if (params.componentIndex == 2){
            let sm = params.data.date_list[0];
            let em = params.data.date_list[params.data.date_list.length-1];
            window.open('/person/' + person_name + "/loc/" + params.name + "/start/" + sm + "/end/" + em);
          }
        });

      },

      processItins(capital, itins_pm){
        let itinFrames = new Array();
        let month_list = new Array();
        let value_map = new Map();
        let date_map = new Map();

        for (let month_i = 0; month_i < itins_pm.length; month_i++) {
          let month_itins = itins_pm[month_i]["itin_list"];
          month_list.push(itins_pm[month_i]["month"]);
          let itinFrame = new Array();
          for (let itin_i = 0; itin_i < month_itins.length; itin_i++) {
            let this_itin = month_itins[itin_i];
            if (!value_map.has(this_itin[0])){
              value_map.set(this_itin[0], 1);
              date_map.set(this_itin[0], new Array(this_itin[1]));
            }else{
              value_map.set(this_itin[0], value_map.get(this_itin[0]) + 1);
              date_map.set(this_itin[0], date_map.get(this_itin[0]).concat([this_itin[1]]));
            }
          }

          for (let loc of value_map.keys()){
            let link = [{'name': capital},
            {'name': loc, 'value': value_map.get(loc), 'date_list': date_map.get(loc)}];
            itinFrame.push(link);
          }

          itinFrames.push(itinFrame);
        };

        return [itinFrames, month_list];
      },
    },
    setup() {
      const person_options = [{'value': 'Donald Trump'}, {'value': 'Emmanuel Macron'},
        {'value': 'Joe Biden'}, {'value': 'Angela Merkel'},
        {'value': 'Vladimir Putin'}, {'value': 'Theresa May'},
        {'value': 'Moon Jae-in'}, {'value': 'Kim Jong-un'},
        {'value': 'Mike Pompeo'}, {'value': 'Scott Morrison'},
        {'value': 'Rex Tillerson'}, {'value': 'Justin Trudeau'},
        {'value': 'John Kerry'}, {'value': 'Giuseppe Conte'},
        {'value': 'Jacinda Ardern'}, {'value': 'Imran Khan'},
        {'value': 'Andrzej Duda'}, {'value': 'Ilir Meta'},
        {'value': 'Boris Johnson'}, {'value': 'Hassan Rouhani'},
        {'value': 'Jair Bolsonaro'}, {'value': 'Mauricio Macri'},
        {'value': 'Paolo Gentiloni'}, {'value': 'Volodymyr Zelensky'}
      ];

      return {
        person_options,
        person_name: ref(''),
        month_range: ref(''),
      };

    },
    
}

</script>

<style>
html,
body {
  margin: 0;
  padding: 0;
}

#echartsMap {
    width: 95%;
    height: 75vh;
}

.map-wrapper {
    background-color: #ffffff00;
    display: flex;
    width: 100%;
    height: 75vh;
    align-items: center;
    justify-content: center;
}
/* element plus */
.demo-date-picker {
  display: flex;
  width: 100%;
  padding: 0;
  flex-wrap: wrap;
}
.demo-date-picker .block {
  padding: 10px 0;
  text-align: center;
  border-right: solid 1px var(--el-border-color-base);
  flex: 1;
}

.demo-date-picker .demonstration {
  display: block;
  color: var(--el-text-color-secondary);
  font-size: 30px;
  margin-bottom: 10px;
}

.el-row{
  justify-content:center;
  align-items:center;
}

.el-col {
  max-width: 22%
}
</style>
