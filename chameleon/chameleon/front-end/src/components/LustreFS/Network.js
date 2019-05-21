import React, { Component } from 'react';
import NetworkGraph from './NetworkGraph';
import * as service from '../../services/getLustreFS';

  class Network extends Component{

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
        console.error('network graph error',error, info);
        this.setState({ hasError: true });
    }

    getGPUMetricList = async () => {
      let time = new Date().getTime();
      let startTime = time - 1000000;
      let endTime = time;

      let network_item_list = "RX_Data,TX_Data";
      // let network_item_list = "asdasdasdasdasdasdasd"; // error test
      //let network_item_list = "RX_Pkts,TX_Pkts,RX_Data,TX_Data,RX_Errs,TX_Errs,RX_Over,TX_Coll";

      const network_metric_result = await service.new_getLustreMetricData( network_item_list ,startTime,endTime);
      const network_metric_result_list = network_metric_result.data.metrics;

      this.network_metric = network_metric_result_list;

      console.log("network" , network_metric_result_list);

      if( typeof network_metric_result_list !== 'undefined' && network_metric_result_list !== null){
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
        if (Object.keys(this.network_metric).length !== 0) {
          return(
            <div className="LustreFS">
              <div className="right_wrap pull-left"  style={{ marginTop : "30px" }} index="2">
                <div className="title_wrap">
                  <div className="text">Network</div>
                </div>
                <div className="cluster_wrap">
                  <div className="graph_wp" >
                    <NetworkGraph
                      mds_metric={this.network_metric}
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

  export default Network;
