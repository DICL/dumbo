import React, { Component } from 'react';
import GPUGraph from './GPUGraph';
import * as service from '../../services/getLustreFS';


  class GPU extends Component{

    constructor(props){
      super(props);
      this.state = {
        isData:false,
        hasError: false
      };
    }

    componentDidMount(){
      this.interval = setInterval(() => this.getGPUMetricList(), 10000)
      // this.getGPUMetricList()
    }

    componentWillUnmount(){
      clearInterval(this.interval);
    }

    componentDidUpdate(){

    }
    componentDidCatch(error, info) {
      // Display fallback UI
      this.setErrorToReport(error, info);

    }

    setErrorToReport = (error, info) => {
        console.error('gpu graph error',error, info);
        this.setState({ hasError: true });
    }

    getGPUMetricList = async () => {
      let time = new Date().getTime();
      let startTime = time - 1000000;
      let endTime = time;

      let gpu_item_list = "PWR,TEMP,SM,MEM,ENC,DEC,MCLK,PCLK";
      // let gpu_item_list = "asdasdasdasd"; // error test

      const gpu_metric_result = await service.new_getLustreMetricData( gpu_item_list ,startTime,endTime);
      const gpu_metric_result_list = gpu_metric_result.data.metrics;

      this.gpu_metric = gpu_metric_result_list;

      console.log("gpu" , gpu_metric_result_list);

      if( typeof gpu_metric_result_list !== 'undefined' && gpu_metric_result_list !== null){
        this.setState({
          isData:true
        });
      }

    }

    render() {
      if (!this.state.isData || this.state.hasError) {
        // Render loading state ...
        return (
          <div></div>
        )
      } else {
        if (Object.keys(this.gpu_metric).length !== 0) {
          return(
            <div className="LustreFS">
              <div className="right_wrap pull-left"  style={{ marginTop : "30px" }} index="2">
                <div className="title_wrap">
                  <div className="text">GPU</div>
                </div>
                <div className="cluster_wrap">
                  <div className="graph_wp" >
                    <GPUGraph
                      mds_metric={this.gpu_metric}
                    />
                  </div>
                </div>
              </div>
            </div>
         );
        }else{
          return (
            <div></div>
          )
        }

      }
    }

  }

  export default GPU;
