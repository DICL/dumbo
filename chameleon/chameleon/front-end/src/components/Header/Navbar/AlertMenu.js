import React from 'react'
import icon_alerts from './img/icon_alerts.png';
import { Dropdown,Card } from 'semantic-ui-react'
import './AlertMenu.css';

const AlertMenu = ({alert_list}) => {

  const trigger = (
    <li className="nav_icon alerts">
      <img src={icon_alerts} alt="alerts"/>
    </li>
  )

  const options = alert_list.map(({Alert})=>{
    return {
      key : Alert.id,
      text : (
        <Card className={ (Alert.state === "CRITICAL") ? "red" : (Alert.state === "WARNING") ? "yellow" : "grey" }>
          <Card.Content>
            <Card.Header content={<p style={{ 'whiteSpace' : 'nowrap', 'width':'270px','textOverflow':'ellipsis','overflow': 'hidden'}}>{Alert.label}</p>} />
            <Card.Meta content={Alert.state} />
            <Card.Description content={<p style={{ 'whiteSpace' : 'pre-wrap', 'width':'270px','wordBreak':'break-all'}}>{Alert.text}</p>} />
          </Card.Content>
        </Card>
      ),
      // disabled: true,
    }
  });

  //console.log(options);


  return (
    <Dropdown trigger={trigger} options={options} icon={null} simple item className="AlertMenu" />
  )
}
/*
<li className="nav_icon alerts">
  <img src={icon_alerts} alt="alerts"/>
</li>

*/
export default AlertMenu
