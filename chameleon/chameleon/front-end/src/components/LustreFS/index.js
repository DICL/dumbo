import React, { Component } from 'react';
import LustreFS from './LustreFS';
import * as service from '../../services/getLustreFS';
import _ from 'lodash';
import Loading from '../Loading';
import GPU from './GPU';
import Network from './Network';

  class LustreMetric extends Component{

    constructor(props){
      super(props);
      this.state = {
        isData:false,
      };
    }

    componentDidMount(){
      //this.getLustreMetricList();
      this.updateChart();
    }

    componentWillUnmount(){
      clearInterval(this.interval);
      this.mds_metric = null
      this.oss_metric = null
    }

    componentDidUpdate(){

    }


    getLustreMetricList = async () => {
      // const metric = await service.getLustreMetricData();

      // let time = Math.round(new Date().getTime()/1000);
      let time = new Date().getTime();
      let startTime = time - 1000000;
      let endTime = time;


      // const metric = await service.getLustreMetricData(startTime,endTime);
      // const metric_list = metric.data.metrics;
      // this.mds_metric = _.pickBy(metric_list,( metric_data, metric_name )=>{ return metric_name.indexOf("Oss") === -1 });
      // this.oss_metric = _.pickBy(metric_list,( metric_data, metric_name )=>{ return metric_name.indexOf("Oss") !== -1 });

      const metric_columns = await service.getLustreMetrics();


      const metric_column_list = metric_columns.data;

      const mds_metric_list = metric_column_list.mds;
      const oss_metric_list = metric_column_list.oss;

      //console.log(metric_column_list);

      this.mds_metric = {};

      const mds_metric_result = await service.new_getLustreMetricData( _.join(mds_metric_list,',') ,startTime,endTime);
      const mds_metric_result_list = mds_metric_result.data.metrics;

      this.mds_metric = mds_metric_result_list;

      // for (var i = 0; i < mds_metric_list.length; i++) {
      //   let temp_metcirname = mds_metric_list[i];
      //   let temp_data = await service.new_getLustreMetricData(temp_metcirname,startTime,endTime)
      //   //console.log(temp_data);
      //   try {
      //     this.mds_metric[temp_metcirname] = temp_data.data.metrics[temp_metcirname];
      //   } catch (e) {
      //     this.mds_metric[temp_metcirname] = 0;
      //   }
      //
      // }

      this.oss_metric = {};

      const oss_metric_result = await service.new_getLustreMetricData( _.join(oss_metric_list,',') ,startTime,endTime);
      const oss_metric_result_list = oss_metric_result.data.metrics;

      this.oss_metric = oss_metric_result_list;

      // for (var j = 0; j < oss_metric_list.length; j++) {
      //   let temp_metcirname = oss_metric_list[j];
      //   let temp_data = await service.new_getLustreMetricData(temp_metcirname,startTime,endTime);
      //   //console.log(temp_data);
      //   try {
      //     this.oss_metric[temp_metcirname] = temp_data.data.metrics[temp_metcirname];
      //   } catch (e) {
      //     this.oss_metric[temp_metcirname] = 0;
      //   }
      //
      // }



      //console.log(this.mds_metric,this.oss_metric);

      this.setState({
        isData:true
      });

      //console.log(this.mds_metric,this.oss_metric);
    }


    updateChart(){
      this.interval = setInterval(() => this.getLustreMetricList(), 3000)
    }


    render() {
      if (!this.state.isData) {
        // Render loading state ...
        return (
          <Loading />
        )
      } else {
        var endTime = new Date().getTime();
        //console.log('loading time',endTime - startTime);
        return(
          <div>
            <LustreFS
              mds_metric={this.mds_metric}
              oss_metric={this.oss_metric}
              />
            <GPU />
            <Network />
          </div>
       );
      }
    }

  }

  export default LustreMetric;
