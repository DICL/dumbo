import React, { Component } from 'react';
import {getYarnJobHistory,getMetricList} from '../../../../../services/getStatus';
import _ from 'lodash';
import PlotlyJS from './PlotlyJS';
import Loading from '../../../../Loading';
import ContainerIdForPIDTable from './ContainerIdForPIDTable';

class YarnJobHistory extends Component {

  constructor() {
    super();
    this.state = {
      isData: false,
    };
  }

  getYarnJobHistory = async ()=>{
    const {idx,  onRemove} = this.props;

    const getMetricList_result = await getMetricList();
    const getYarnJobHistory_result = await getYarnJobHistory(this.props.application_id,this.props.start_time);
    const application_list = {
      data : _.filter(getYarnJobHistory_result.data,(item)=>{return item.application_id === this.props.application_id})
    }
    const metric_registry_list = getMetricList_result.data;
    //const origenData = this.origenData = application_list.data;
    const container_id_list = _.uniq(_.map(application_list.data,(item)=>{return item.container_id}));

    // const metric_list = _.filter(_.keys(data_list[0]),(item)=>{return !defalut_keys.includes(item)})
    const defalut_keys = ['application_id','container_id','create_date','node','pid']



    // 메트릭 리스트
    const metric_list =
    _.filter( // 4. 3에서 나온 결과중에서 defalut_keys 하고 일치 하지 않는 것들만 추출
      _.uniq( // 3. 해당 리스트 중에서 유니크한것들만 추출
        _.flattenDeep(  // 2. 1 에서 나온 리스트를 가지고 평탄화
          _.map(   // 1. 전체리스트를 가져와서 키값만 추출하여 맵핑
            application_list.data,
            (item)=>{ return _.keys(item); }
          )
        )
      ),
      (item)=>{
        return !defalut_keys.includes(item)
      }
    );



    let index = 0;
    // console.log(this.props);
    //console.log(application_list,container_id_list);

    /* #########################################
     *   PlotlyJS 그리기 위한 데이터 정렬
     *  1. container_id 별로 데이터 정렬
    ############################################# */
    this.applications = {}

    _.each(container_id_list,(container_id)=>{
      this.applications[container_id] = {};
      let tmpList = _.filter(application_list.data,(filter_target)=>{return filter_target.container_id === container_id});

      this.applications[container_id]['node'] = _.get(_.uniq(_.map(tmpList,(item)=>{return item.node})),0);
      this.applications[container_id]['pid'] = _.get(_.uniq(_.map(tmpList,(item)=>{return item.pid})),0);
      this.applications[container_id]['max'] = application_list.data[application_list.data.length - 1]['create_date'];
      this.applications[container_id]['min'] = application_list.data[0]['create_date'];

      //let metric_list = _.uniq(_.map(tmpList,(item)=>{return item.metric}));
      // const defalut_keys = ['application_id','container_id','create_date','node','pid']
      // let metric_list = _.filter(_.keys(tmpList[0]),(item)=>{return !defalut_keys.includes(item)})
      //console.log(metric_list);


      this.applications[container_id]['metric'] = {};

      _.each(metric_list,(metric_name)=>{

        this.applications[container_id]['metric'][metric_name] = {};

        let temp_metric_list = _.filter(application_list.data,(filter_target)=>{return filter_target.container_id === container_id });
        // je.kim 생성일자 별로 재정렬
        temp_metric_list = _.orderBy(temp_metric_list, (order_metric_item)=>{ return new Date(order_metric_item.create_date); }, ['asc' ]);
        //console.log(temp_metric_list);
        // je.kim 서버의 시간은 동부 표준시로 16시간 늦음
        // 반대로 디비에 저장되었는 시간은 표준시로 9시간 늦음
        // -> 5시간 더해야함
        // 190408 je.kim 김재억아 디비에 적재할때 표준시로 해서 적재하면 되는걸 왜 5시간을 더해서 문재를 발생시키냐?
        this.applications[container_id]['metric'][metric_name]['x'] = _.map(temp_metric_list,(metric_list)=>{let temp_time=  new Date(metric_list.create_date);  /*temp_time.setHours(temp_time.getHours() + 5);*/ return temp_time; });
        this.applications[container_id]['metric'][metric_name]['y'] = _.map(temp_metric_list,(metric_list)=>{return parseFloat(metric_list[metric_name])});
        this.applications[container_id]['metric'][metric_name]['data'] = temp_metric_list;
        this.applications[container_id]['metric'][metric_name]['index'] = index ++ ;
      });




      //const host_list = _.chain(origenData).map((indexData)=>{return indexData.node}).sort().uniq().value();
    })

    /* #########################################
     *   PlotlyJS 그리기 위한 데이터 정렬
     *  2. metric 별로 데이터 정렬
    ############################################# */
    let set_data = [];
    _.each(this.applications,(container_item , container_name)=>{
      let pid = container_item.pid;
      _.each(container_item.metric, (metric_item , metric_name)=>{
        set_data.push({
          x: metric_item.x,
          y: metric_item.y,
          //mode: 'lines',
          type: 'scatter',
          metric_name : metric_name,
          // 11.26 je.kim [pid metric_name] -> [node pid]
          name: `${pid}@${container_item.node}`,
          //name: `${pid} ${metric_name}`,
          //line: {color: randomColor()}
        });
      })
    });
    this.set_data = set_data;

    //console.log(application_list.data);
    //console.log(this.applications);
    this.setState({
      isData: true,
      metric_list : metric_list,
      metric_registry_list: metric_registry_list,
    });



    console.log("container_id ==>",this.applications,"plotly js =>",this.set_data, 'getMetricList_result==>' , metric_registry_list);
    // 데이터가 없을경우 modal 창닫기
    if(this.set_data.length === 0){
      alert('Not Find Data');
      onRemove(idx);
      return;
    }
  }

  // 11.26 je.kim 상단 메뉴 제거
  //


  componentDidMount(){
    this.getYarnJobHistory()
  }

  render () {

    if (!this.state.isData) {
      return(<Loading />)
    }else{
      // return(
      //   <div>
      //
      //     <PlotlyJS
      //       data={this.set_data}
      //       application_id={this.props.application_id}
      //
      //       />
      //   </div>
      //
      // );
      const result =  _.map(this.state.metric_list,(metric_name,index)=>{
      let tmp_list =
       _.filter(
         this.set_data,
         (plotlyJS_line_data)=>{
           return plotlyJS_line_data.metric_name === metric_name;
         }
       );

       let find_metric_registry =
        _.find(
          this.state.metric_registry_list,
          (metric_registry_item)=>{
            return metric_registry_item.col_name === metric_name
          }
        );
        let title = typeof find_metric_registry === 'undefined' ? metric_name : find_metric_registry['name']
        let y_label = typeof find_metric_registry === 'undefined' ? undefined : find_metric_registry['y_axis_label']
       return(
         <div
           key={index}
           className='graph_area'
           >
             <PlotlyJS
               y_label={y_label}
               title={`${title}`}
               data={tmp_list}
               application_id={this.props.application_id}
               />
             <ContainerIdForPIDTable
              container_list={ _.map(this.applications,(app_data,container_id) => { return { 'container_id' : container_id , 'pid' : app_data.pid } }) }
             />
         </div>
       );
     })
      return (
        <div>
          {result}
        </div>
      )
    }

  }
}


export default YarnJobHistory
