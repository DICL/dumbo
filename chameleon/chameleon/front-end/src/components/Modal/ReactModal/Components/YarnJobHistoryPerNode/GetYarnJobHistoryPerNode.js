import React, { Component } from 'react';
import {getYarnJobHistoryPerNode,getMetricList} from '../../../../../services/getStatus';
import {randomColor} from '../../../../../services/utils';
import _ from 'lodash';
import PlotlyJS from './PlotlyJS';
import Loading from '../../../../Loading';

class GetYarnJobHistoryPerNode extends Component{
    pad = function (n, width) {
      n = n + '';
      return n.length >= width ? n : new Array(width - n.length + 1).join('0') + n;
    }
    dateConverterToString = (datetime)=>{
      var a = datetime;
      //var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      var year = a.getFullYear();
      //var month = months[a.getMonth()];
      var date = a.getDate();
      var hour = a.getHours();
      var min = a.getMinutes();
      var sec = a.getSeconds();
      var time = year +  this.pad((a.getMonth()+1),2) +  this.pad(date,2) +   this.pad(hour,2) +  this.pad(min,2) +  this.pad(sec,2)  ;
      // var time = date + ' ' + month + ' ' + year + ' ' + hour + ':' + min + ':' + sec ;
      return parseInt(time,10);
    };



    getYarnJobHistoryPerNode = async ()=>{
      const {idx,  onRemove} = this.props;
      const {node, start_time, end_time} = this.props.data
      const defalut_keys = ['application_id','container_id','create_date','node','pid']
      let send_data = {
        node : node,
        start_time : this.dateConverterToString(start_time),
        end_time : this.dateConverterToString(end_time),
      }
      const getMetricList_result = await getMetricList();
      const metric_registry_list = getMetricList_result.data;
      const getYarnJobHistoryPerNode_result = await getYarnJobHistoryPerNode(send_data);

      // 해당 노드만 추출
      const data_list = _.filter(getYarnJobHistoryPerNode_result.data,(item)=>{ return item.node === node });
      //const metric_list = _.filter(_.keys(data_list[0]),(item)=>{return !defalut_keys.includes(item)})
      // 1. 전체리스트를 가져와서 키값만 추출하여 맵핑
      // 2. 1 에서 나온 리스트를 가지고 평탄화
      // 3. 해당 리스트 중에서 유니크한것들만 추출
      // 4. 3에서 나온 결과중에서 defalut_keys 하고 일치 하지 않는 것들만 추출
      const metric_list = _.filter(_.uniq(_.flattenDeep(_.map(data_list,(item)=>{ return _.keys(item); }))),(item)=>{return !defalut_keys.includes(item)});

      // const application_list =
      // _.uniq( // 해당 application_id 을 토대로 유니크한 값만 추출
      //   _.map( // 전체 리스트 중에서 application_id 만 추출
      //     data_list,
      //     (item)=>{
      //       return item.application_id;
      //     }
      //   )
      // )

      let metric_for_graph_list =
        // 메트릭 이름별로 리스트 생성
        _.map(
          metric_list,
          (metric_name,matric_for_index)=>{
            // 해당 메트릭 이름이 존재하는 appid 리스트 추출
            let temp_appidlist_filter_for_metric =
              _.uniq( // 해당 결과물에서 유니크한 값만 추출
                _.map( // 추출된 리스트 중에서 application_id 만 추출
                  _.filter( // 전체 리스트중에서 메트릭이 undefined 이 아닌 리스트 추출
                    data_list,
                    (filter_target)=>{
                      return typeof filter_target[metric_name] !== 'undefined';
                    }
                  ),
                  (item)=>{
                    return item.application_id;
                  }
                )
              )
            //  console.log(temp_appidlist_filter_for_metric);
            // 메트릭 이름을 기준으로 리스트 생성
            return {
              metric_name : metric_name,
              grape_data :
                _.map(
                temp_appidlist_filter_for_metric,
                (application_id)=>{
                  // 해당 application_id 별로 필터링
                  let filter_list_for_application_id =
                  _.filter(
                    data_list,
                    (filter_target)=>{
                      return filter_target.application_id === application_id;
                    }
                  )

                  // 필터링 된 리스트를 생성일자 별로 재정렬
                  filter_list_for_application_id = _.orderBy(filter_list_for_application_id, (order_metric_item)=>{ return new Date(order_metric_item.create_date); }, ['asc' ]);

                  // 정렬된 내용을 기준으로 일자별로 합산하여 리스트 생성
                  let avg_for_datetime = {}
                  _.each(filter_list_for_application_id,(data)=>{
                    if(avg_for_datetime[data.create_date]){
                      let tmp_data = avg_for_datetime[data.create_date];

                      let tmp_number = _.toNumber(data[metric_name]);
                      tmp_number = ! _.isNaN(tmp_number) ? tmp_number : 0;
                      tmp_data = tmp_data[metric_name] + tmp_number;

                      avg_for_datetime[data.create_date] = tmp_data;
                    }else{
                      avg_for_datetime[data.create_date] = null
                      let tmp_number = _.toNumber(data[metric_name]);
                      tmp_number = ! _.isNaN(tmp_number) ? tmp_number : 0;
                      avg_for_datetime[data.create_date] = tmp_number;
                    }
                  });

                  console.log('test1',avg_for_datetime);

                  return {
                    // x : _.map(filter_list_for_application_id,(item)=>{ let temp_time=  new Date(item.create_date);  temp_time.setHours(temp_time.getHours() + 5);  return item.create_date }),
                    // y : _.map(filter_list_for_application_id,(item)=>{ return typeof item[metric_name] === 'undefined' ? null : item[metric_name] }),
                    // 190408 je.kim 김재억아 디비에 적재할때 표준시로 해서 적재하면 되는걸 왜 5시간을 더해서 문재를 발생시키냐?
                    x : _.map(_.keys(avg_for_datetime),(item)=>{ let temp_time=  new Date(item);  /* temp_time.setHours(temp_time.getHours() + 5); */ return temp_time }),
                    y : _.map(avg_for_datetime,(item)=>{ return item }),
                    name: `${application_id}`,
                    type: 'bar',
                    application_id : application_id,

                  };
                }
              )
            }
          }
        )
        console.log(metric_for_graph_list);


      let avg_for_datetime = {};

      _.each(data_list,(data)=>{
        if(avg_for_datetime[data.create_date]){
          let tmp_data = avg_for_datetime[data.create_date];
          _.each(metric_list,(metric_name)=>{
            if(typeof data[metric_name] !== undefined){
              let tmp_number = _.toNumber(data[metric_name]);
              tmp_number = ! _.isNaN(tmp_number) ? tmp_number : 0;
              tmp_data[metric_name] = tmp_data[metric_name] + tmp_number;
            }
          })
          avg_for_datetime[data.create_date] = tmp_data;
        }else{
          avg_for_datetime[data.create_date] = {}
          _.each(metric_list,(metric_name)=>{
            if(typeof data[metric_name] !== undefined){
              let tmp_number = _.toNumber(data[metric_name]);
              tmp_number = ! _.isNaN(tmp_number) ? tmp_number : 0;
              avg_for_datetime[data.create_date][metric_name] = tmp_number;
            }
          })
        }
      });
      let tmp_list  = _.map(avg_for_datetime, (value,key)=>{value['create_date'] = key; return value;})

      let grape_data = _.map(metric_list,(metric_name)=>{
        let tmp_metric_filter = _.filter(tmp_list,(item)=>{ return ( !_.isUndefined(item[metric_name]) ) });
        return {
          x: _.map(tmp_metric_filter,(item)=>{let temp_time=  new Date(item.create_date);  temp_time.setHours(temp_time.getHours() + 5);  return item.create_date}),
          y: _.map(tmp_metric_filter,(item)=>{return item[metric_name]}),
          // //mode: 'lines',
          //type: 'scatter',
          type: 'bar',
          metric_name:metric_name,
          name: `${metric_name}`,
          line: {color: randomColor()}
        }
      })

      if(data_list.length === 0){
        alert('Not Find Data');
        onRemove(idx);
        return;
      }

      console.log('send_data=>',send_data,'recive_data=>',data_list,'metric_list=>',metric_list,'group_by_datetime=>',avg_for_datetime,'grid_data=>',grape_data);
      this.setState({
        isData : true,
        grape_data : grape_data,
        //metric_list : metric_list,
        metric_for_graph_list:metric_for_graph_list,
        metric_registry_list: metric_registry_list,
      });
    }

    constructor(props){
      super(props);
      this.state = {
        isData : false,
        grape_data: null,
      }
    }

    componentDidMount(){
      this.getYarnJobHistoryPerNode();
    }

    componentWillUnmount(){

    }

    componentDidUpdate(){

    }

    render() {
      if (!this.state.isData) {
        // return(<div>test....</div>)
        return(<Loading />)
      }else{
        let tmp_list = _.map(
          this.state.metric_for_graph_list
          ,(matric_graph,index)=>{
            let metric_name = matric_graph.metric_name;
            let find_metric_registry =
             _.find(
               this.state.metric_registry_list,
               (metric_registry_item)=>{
                 return metric_registry_item.col_name === metric_name
               }
            );
            let title = typeof find_metric_registry === 'undefined' ? metric_name : find_metric_registry['name'];
            let y_label = typeof find_metric_registry === 'undefined' ? undefined : find_metric_registry['y_axis_label']
            return(
              <div
                  className="graph_area"
                  key={index}
                  >
                  <PlotlyJS
                    title={`${title}`}
                    y_label={y_label}
                    grape_data={matric_graph.grape_data}
                    data = {{
                      min : this.props.data.start_time,
                      max : this.props.data.end_time
                    }}
                    />
              </div>
            );
          }
        )


        // let tmp_list = _.map(
        //   this.state.metric_list
        //   ,(metric_name,index)=>{
        //     let temp_grape_data_for_metric =
        //     _.filter(
        //       this.state.grape_data
        //       ,(grape_data_item)=>{
        //         return grape_data_item.metric_name === metric_name
        //       }
        //     );
        //
        //     let find_metric_registry =
        //      _.find(
        //        this.state.metric_registry_list,
        //        (metric_registry_item)=>{
        //          return metric_registry_item.col_name === metric_name
        //        }
        //      );
        //     let title = typeof find_metric_registry === 'undefined' ? metric_name : find_metric_registry['name'];
        //     let y_label = typeof find_metric_registry === 'undefined' ? undefined : find_metric_registry['y_axis_label']
        //     //console.log(find_metric_registry);
        //     // let title = metric_name;
        //     return (
        //       <div
        //         className="graph_area"
        //         key={index}
        //         >
        //         <PlotlyJS
        //           title={`${title}`}
        //           y_label={y_label}
        //           grape_data={temp_grape_data_for_metric}
        //           data = {{
        //             min : this.props.data.start_time,
        //             max : this.props.data.end_time
        //           }}
        //           />
        //       </div>
        //
        //     )
        //   }
        // )

        // return(
        //   <div>
        //     <PlotlyJS
        //       grape_data={this.state.grape_data}
        //       data = {{
        //         min : this.props.data.start_time,
        //         max : this.props.data.end_time
        //       }}
        //       />
        //   </div>
        //
        // );
        return(<div>{tmp_list}</div>);
      }
    }

  }

export default GetYarnJobHistoryPerNode;
