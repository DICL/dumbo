import React, { Component } from 'react';
import Menu, {SubMenu, MenuItem} from 'rc-menu';
import { Checkbox } from 'semantic-ui-react'
import 'rc-menu/assets/index.css';


class Select extends Component{
  toggle = () => this.setState({ checked: !this.state.checked })
  render() {
    const { application_id,SubHostsMenu,changeToggle } = this.props;
    //console.log(SubHostsMenu);
    return (
      <Menu
        mode="horizontal"
        multiple={false}
        selectable={false}
         >
        <SubMenu title="Applications">
            <SubMenu title={application_id}>
              {
                SubHostsMenu.map( (host_info, depth1_index) => {
                  return <SubMenu key={`${depth1_index}`} title={<Checkbox label={host_info.host_name} checked={host_info.checked} onChange={(e)=>{
                      changeToggle('host',{host_name:host_info.host_name},!host_info.checked)
                    }}/>}>
                    {
                      host_info.container_list.map((container_info,depth2_index)=>{
                        return <SubMenu key={`${depth1_index}-${depth2_index}`} title={<Checkbox label={container_info.container_name} checked={container_info.checked} onChange={(e)=>{
                            changeToggle('container',{host_name:host_info.host_name,container_id:container_info.container_id},!container_info.checked)
                          }}/>}>
                          {
                            container_info.metric.map((metric_info,depth3_index)=>{
                              return <MenuItem key={`${depth1_index}-${depth2_index}-${depth3_index}`}><Checkbox label={metric_info.metric_name} checked={metric_info.checked} onChange={(e)=>{
                                  changeToggle('metric',{host_name:host_info.host_name,container_id:container_info.container_id,metric:metric_info.metric_name},!metric_info.checked)
                                }}/></MenuItem>
                            })
                          }
                        </SubMenu>
                        //return <MenuItem key={`${depth1_index}-${depth2_index}`}><Checkbox label={container_info.container_id} /></MenuItem>
                      })
                    }
                  </SubMenu>
                  //return <MenuItem key={`${depth1_index}`}><Checkbox label={host_info.host_name} /></MenuItem>
                })
              }
            </SubMenu>
        </SubMenu>
      </Menu>
    );
  }
}

Select.defaultProps = {
    SubHostsMenu: [],
    changeToggle: (type,target,set_checked)=>{
      console.log(type,target,set_checked);
    },
};

/*

<Menu mode="horizontal" >
  <SubMenu title="Applications">
      <SubMenu title={application_id}>
        {SubHostsMenu}
      </SubMenu>
  </SubMenu>
</Menu>


  <Menu mode="horizontal" >
    <SubMenu title="Applications">
      <SubMenu title="application_1536191979328_0127">
        <SubMenu title={<Checkbox label='node01' />}>
          <SubMenu title={<Checkbox label='container_e03_1536191979328_0127_01_000002' />}>
              <MenuItem><Checkbox label='cpu_used' /></MenuItem>
          </SubMenu>
          <SubMenu title={<Checkbox label='container_e03_1536191979328_0127_01_000003' />}>
              <MenuItem><Checkbox label='cpu_used' /></MenuItem>
          </SubMenu>
        </SubMenu>
        <SubMenu title={<Checkbox label='node02' />}>
          <SubMenu title={<Checkbox label='container_e03_1536191979328_0127_01_000001' />}>
            <MenuItem><Checkbox label='cpu_used' /></MenuItem>
          </SubMenu>
        </SubMenu>
      </SubMenu>
    </SubMenu>
  </Menu>
*/



export default Select;
