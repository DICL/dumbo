import React from 'react';
// import OSSGrape from './OSSGrape';
import NewOSSGrape from './NewOSSGrape';
import _ from 'lodash';



const OSS = ({oss_metric}) => {
  //console.log(oss_metric);

  // OSS 서버 리스트 그리기
  let oss_name_list =
    // 체인설정
    _.chain(oss_metric)
    // 키값들을 추출 Oss1-lustre0-Read
    .keys()
    // OSS1만 추출
    .map( (oss_metric_name) => { return ( oss_metric_name.split('-').length > 1 ) ? oss_metric_name.split('-')[0] : '';})
    // 내용이 없는 리스트들을 제거
    .filter( (oss_metric_name) => { return oss_metric_name !== '' } )
    // 중복값들을 제거
    .uniq()
    // 순차대로 분류
    .sortBy( (oss_metric_name) => { return oss_metric_name } )
    // 결과 출력
    .value();

  // 러스터 파일시스템별로 분류
  let lustre_fs_list =
  _.chain(oss_metric)
    .keys()
    .map( (oss_metric_name) => { return ( oss_metric_name.split('-').length > 1 ) ? oss_metric_name.split('-')[1] : '';})
    .filter( (oss_metric_name) => { return oss_metric_name !== '' } )
    .uniq()
    .value();
  //console.log(oss_name_list,lustre_fs_list);

  let set_metric = [];

  // OSS 그래프 그리기
  _.each(
    oss_name_list,
    (oss_name)=>{
      var set_data = [];
      _.each(
        lustre_fs_list,
        (lustre_fs_name) =>{
          let search_read_name = `${oss_name}-${lustre_fs_name}-Read`;
          let search_write_name = `${oss_name}-${lustre_fs_name}-Write`;
            set_data.push({
              x: _.map(oss_metric[search_read_name],(data)=>{ return (data[1] ) }),
              y: _.map(oss_metric[search_read_name],(data)=>{ return data[0] }),
              mode: 'lines',
              type: 'scatter',
              name: search_read_name,
            });
            set_data.push({
              x: _.map(oss_metric[search_write_name],(data)=>{ return (data[1] ) }),
              y: _.map(oss_metric[search_write_name],(data)=>{ return data[0] }),
              mode: 'lines',
              type: 'scatter',
              name: search_write_name,
            });
        }
      )
      set_metric.push(
          <NewOSSGrape
            key={oss_name}
            title={oss_name}
            data={set_data}
            />
        )
    }
  );


  // let metrics_length = _.keys(oss_metric).length / 2;


  // for (var i = 1; i <= metrics_length; i++) {
  //   let temp_name = `Oss${i}`;
  //
  //   let search_read_name = `${temp_name}Read`;
  //   let search_write_name = `${temp_name}Write`;
  //
  //   let set_data = [
  //     {
  //       x: _.map(oss_metric[search_read_name],(data)=>{ return (data[1] ) }),
  //       y: _.map(oss_metric[search_read_name],(data)=>{ return data[0] }),
  //       mode: 'lines',
  //       type: 'scatter',
  //       name: search_read_name,
  //     },
  //     {
  //       x: _.map(oss_metric[search_write_name],(data)=>{ return (data[1] ) }),
  //       y: _.map(oss_metric[search_write_name],(data)=>{ return data[0] }),
  //       mode: 'lines',
  //       type: 'scatter',
  //       name: search_write_name,
  //     }
  //   ]
  //   // let temp_metric[temp_name] = {}
  //   //
  //   //
  //   //
  //   // temp_metric[temp_name][search_read_name] = oss_metric[search_read_name]
  //   // temp_metric[temp_name][search_write_name] = oss_metric[search_write_name]
  //
  //   set_metric.push(
  //     <NewOSSGrape
  //       key={temp_name}
  //       title={temp_name}
  //       data={set_data}
  //       />
  //   )
  // }


//   let result = _.map(oss_metric,(metric_data,metric_name)=>{
//
//     let tmp_data = {
//       title : metric_name,
//       set_data : {
//         x: _.map(metric_data,(data)=>{ return (data[1] ) }),
//         y: _.map(metric_data,(data)=>{ return data[0] }),
//         mode: 'lines',
//         type: 'scatter',
//         name: `${metric_name}`,
//       }
//     }
//
//     return(
//       <OSSGrape
//         key={metric_name}
//         data={tmp_data}
//         />
//     )
//
//   }
//
//
// )

  return(
    <div className="ui three two grid">
      {/* {result} */}
      {set_metric}
    </div>
 );
}

export default OSS;
