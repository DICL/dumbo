import React, { Component } from 'react';
import _ from 'lodash';

import Menu, {SubMenu, MenuItem} from 'rc-menu';
import { Checkbox } from 'semantic-ui-react'
import 'rc-menu/assets/index.css';

import './Select.css';

class Select extends Component{


  render() {
    const {data , viewOrHideTraces} = this.props;
    const application_list =
      _.chain(data)
      .map((container_info)=>{return container_info.application_id})
      .uniq()
      .value()
    ;

    //const host_list = _.chain(data).map((container_info)=>{return container_info.node}).sort().uniq().value();

    let menus = _.map(application_list,(application_id)=>{
      return {
        "application_id" : application_id,
        "host_list" : _.chain(data)
        .filter((target)=>{return target.application_id === application_id})
        .map((indexData)=>{return indexData.node})
        .sort().uniq().value()
        .map((host_name)=>{
          return {
            "host_name" : host_name,
            "container_list" : _.chain(data)
              .filter((container_info)=>{return container_info.node === host_name && container_info.application_id === application_id})
              .map((indexData)=>{return indexData.container_id})
              .uniq().sort().value().map(
                (container_id)=>{
                  let contains_info = _.find(data,(item)=>{return item.container_id === container_id});
                  return {
                    container_id : container_id,
                    container_name : `[PID ${contains_info.pid}] ${container_id}`,
                  }
                }
              ),
          }
        }),

      }
    })

    // console.log(menus);
    return(
      <Menu
        mode="horizontal"
        multiple={false}
        selectable={false}
         >
         <SubMenu title="Applications">
             {
               menus.map((application_info,depth1_index)=>{
                 return <SubMenu key={depth1_index} title={application_info.application_id}>
                   {
                     application_info.host_list.map((host_info,depth2_index)=>{
                       return <SubMenu key={`${depth1_index}-${depth2_index}`} title={host_info.host_name}>
                         {
                           host_info.container_list.map((container_info,depth3_index)=>{
                             return <MenuItem key={`${depth1_index}-${depth2_index}-${depth3_index}`}>
                               <Checkbox label={container_info.container_name} defaultChecked onChange={
                                   (e, checked) => { e.stopPropagation(); /* onToggle 이 실행되지 않도록 함 */  viewOrHideTraces(checked,container_info.container_id);  }
                                }/>
                             </MenuItem>
                           })
                         }
                        </SubMenu>
                     })
                   }
                 </SubMenu>
               })
             }
         </SubMenu>
      </Menu>
    );
  }
}


export default Select;
