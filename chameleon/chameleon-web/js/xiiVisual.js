var xiiVisual={};
window.xiiVisual=window.xiiVisual||xiiVisual;

xiiVisual.networkTopo = function(target, gWidth, gHeight, nodeData, routerData, switchData){
                var margin = {top:20, right: 20, bottom: 20, left: 60},
                            width = gWidth - margin.left - margin.right,
                            height = gHeight - margin.top - margin.bottom;


                var color = d3.scale.ordinal().range(["#94CD7A", "#019CCB", "#fff"]);

                var svg = d3.select(target).append("svg")
                    .attr("width", gWidth)
                    .attr("height", gHeight);
                    // .append("g")
                    // .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


                //////////////////////// **** DATA **** ///////////////////////////////

                var nodeColor = ["#8ec0da","#e8f2f8","#8ec0da","#4d9bc4","#8ec0da"];

                var dataContainer = svg.append("g").attr("class", "dataContainer").attr("transform", "translate(" + (gWidth - margin.right - 230) + "," + (margin.top) + ")");


                var rackArr = {};
                var rackObj = {};
                nodeData.forEach(function(e, i){ 
                    if(typeof rackArr[e.rack] != "undefined" ){ rackArr[e.rack]++; } else { rackArr[e.rack] = 1; }
                    if(typeof rackObj[e.rack] != "undefined" ){ 
                       rackObj[e.rack][e.name] = e.info;
                    } else { 
                        rackObj[e.rack] = {};
                        rackObj[e.rack][e.name] = e.info;
                    }
                });

                var rackHeight = 250,
                    rackMargin = 10,
                    tempRackY = 0;


                var rackContainer = dataContainer.selectAll(".rackContainer").data(d3.entries(rackArr)).enter().append("g")
                                    .attr("class", "rackContainer")
                                    .attr("transform", function(d,i){ return "translate(0,"+ getRackY(d,i) +")"; }) 
                                    .attr("rack", function(d,i){ return d.key; });
               
                rackContainer.selectAll(".rackBack")
                              .data(function(d){ return [d]; })
                              .enter()
                              .append("rect")
                              .attr("width", 225)
                              .attr("height", function(d,i){ if(d.value == 1){ return d.value * rackHeight; } else { return d.value * rackHeight - ((d.value-1) *30);  } })
                              .attr("x", 0)
                              .attr("y", function(d,i){ return getRackY(d,i); })
                              .attr("rx", 2).attr("ry", 2)
                              .style({"fill":"#becbd6", "stroke":"#a6b2bc", "stroke-width":"0.4px"});

                rackContainer.selectAll(".rackText")
                              .data(function(d){ return [d]; })
                              .enter().append("text")
                              .attr("x", 110)
                              .attr("y", 30)
                              .text(function(d){ return "Server Rack " + d.key  })
                              .style({"fill":"#000", "font-size":"16px", "text-anchor":"middle", "font-weight":"bold"});



                var nodeContainer = rackContainer.selectAll(".nodeContainer")
                                    .data(function(d){ return d3.entries(rackObj[d.key]); })
                                    .enter().append("g")
                                    .attr("transform", function(d,i){ return "translate(20,"+( i * 210 + 50)+")"; }) 
                                    .attr("class", "nodeContainer")
                                    .attr("name",function(d){ return d.key; });

                nodeContainer.append("rect")
                            .attr("width", 171)
                            .attr("height", 188)
                            .attr("x", 10)
                            .attr("y", 0)
                            .attr("rx", 2).attr("ry", 2)
                            .style({"fill":"#fff", "stroke":"#b7b7b7", "stroke-width":"0.4px"});

                nodeContainer.append("rect")
                            .attr("class","routerConnector")
                            .attr("x",-160)
                            .attr("y",80)
                            .attr("width",140)
                            .attr("height",7)
                            .style("fill",function(d){ return getConnectorColor(d.key, "router"); });

                nodeContainer.append("rect")
                            .attr("class","switchConnector")
                            .attr("x",-200)
                            .attr("y",130)
                            .attr("width",180)
                            .attr("height",7)
                            .style("fill",function(d){ return getConnectorColor(d.key, "switch"); });

                nodeContainer.selectAll(".routerIp")
                            .data(function(d){ return [d.key]; })
                            .enter()
                            .append("text")
                            .attr("x",function(d){ return -80; })
                            .attr("y",function(d){ return 70; })
                            .text(function(d){ return routerData[d];})
                            .style("text-anchor", "middle");

                nodeContainer.selectAll(".switchIp")
                            .data(function(d){ return [d.key]; })
                            .enter()
                            .append("text")
                            .attr("x",function(d){ return -80; })
                            .attr("y",function(d){ return 120; })
                            .text(function(d){ return switchData[d];})
                            .style("text-anchor", "middle"); 


                nodeContainer.selectAll(".nodeInfoBack")
                              .data(function(d) { return d3.entries(d.value);  })
                              .enter().append("rect")
                              .attr("width", 160)
                              .attr("height",23)
                              .attr("x", 15)
                              .attr("y", function(d, i) { return (i * 30) + 39; })
                              .style("fill", function(d,i) { return nodeColor[i]; })
                              .attr("rx", 3).attr("ry", 3);

                nodeContainer.selectAll(".nodeName")
                              .data(function(d) { return [d.key]; })
                              .enter().append("text")
                              .attr("x", 95)
                              .attr("y", 25)
                              .attr("class", "nodeName")
                              .text(function(d,i) { return d; })
                              .style("text-anchor", "middle");
                            
                nodeContainer.selectAll(".nodeInfo")
                              .data(function(d) { return d3.entries(d.value); })
                              .enter().append("text")
                              .attr("x", function(d) { return 95; })
                              .attr("y", function(d, i) { return (i * 30) + 54; })
                              .attr("class", "nodeInfo")
                              .text(function(d,i) { return d.key + " : " + d.value; })
                              .style("text-anchor", "middle");



           

                var routerContainer = svg.append("g").attr("class", "routerContainer");
                var routerConnectorW = 140;

                var routerLineVX = 390;

                var routerInfo = {
                    "image" : "images/test/router.png",
                    "X" : 70,
                    "Y" : 170,
                    "color" : "rgba(238, 89, 82, 1)",
                    "imageW" : 103,
                    "imageH" : 103
                };
                appendRect(routerContainer, "", 26, getRouterInternalH, routerLineVX, getConnectorY("router"), routerInfo.color);
                appendRect(routerContainer, "", 230, 7, routerInfo.imageW + routerInfo.X, routerInfo.Y + routerInfo.imageW*0.5, routerInfo.color);
                appendImage(routerContainer, "routerImage", routerInfo.image, routerInfo.imageW, routerInfo.imageH, routerInfo.X, routerInfo.Y);
                appendText(routerContainer, routerInfo.imageW + 130, (routerInfo.Y + routerInfo.imageW*0.5) - 13, "172.168.1.1", "#364347");
                
                routerContainer.append("text")
                              .attr("x",0)
                              .attr("y",0)
                              .attr("transform", "translate("+(routerLineVX + 18)+","+(getConnectorY("router")+160)+") rotate(-90)")
                              .text("Internal")
                              .style("fill","#fff")
                              .style("font-size","17px");
               
                var switchContainer = svg.append("g").attr("class", "switchContainer");
                var switchConnectorW = 190;

                var switchInfo = {
                    "image" : "images/test/switch.png",
                    "X" : 70,
                    "Y" : 400,
                    "color" : "rgba(70, 134, 199, 1)",
                    "imageW" : 103,
                    "imageH" : 103
                };

                var switchLineVX = 350;

                appendRect(switchContainer, "", 26, getSwitchInternalH, switchLineVX, getConnectorY("switch"), switchInfo.color);

                appendRect(switchContainer, "", 200, 7, switchInfo.imageW + switchInfo.X, switchInfo.Y + switchInfo.imageW*0.5, switchInfo.color);
                appendImage(switchContainer, "switchImage", switchInfo.image, switchInfo.imageW, switchInfo.imageH, switchInfo.X, switchInfo.Y);
                appendText(switchContainer, switchInfo.imageW + 130, (switchInfo.Y + switchInfo.imageW*0.5) - 13, "182.168.1.1", "#364347");

                switchContainer.append("text")
                              .attr("x",0)
                              .attr("y",0)
                              .attr("transform", "translate("+(switchLineVX + 18)+","+(getConnectorY("switch")+230)+") rotate(-90)")
                              .text("Internal")
                              .style("fill","#fff")
                              .style("font-size","17px");



                function getRouterInternalH(){
                  var length = d3.selectAll(target + " .routerConnector")[0].length;
                  var max = 0;
                  var t = d3.selectAll(target + " .routerConnector")[0];
                  for(var i=0; i<t.length; i++){
                    if(t[i].getBoundingClientRect().top > max){ max = t[i].getBoundingClientRect().top; }
                  }
                  return max - t[0].getBoundingClientRect().top + 7;
                }

                function getSwitchInternalH(){
                   var length = d3.selectAll(target + " .switchConnector")[0].length;
                  var max = 0;
                  var t = d3.selectAll(target + " .switchConnector")[0];
                  for(var i=0; i<t.length; i++){
                    if(t[i].getBoundingClientRect().top > max){ max = t[i].getBoundingClientRect().top; }
                  }
                  return max - t[0].getBoundingClientRect().top + 7;
                }
                            
                function getConnectorColor(d, type){
                  var v = 1;
                  v = Math.random().toFixed(1) * 1;
                  if(v == 0){ v = 0.1; }
                  var color = "";
                  type == "router" ? color = "rgba(238, 89, 82, "+v+")" : color = "rgba(70, 134, 199, "+v+")";
                  return color;
                }

                function getConnectorY(type){
                  var targetTop = $(target).offset().top;
                  if(type == "router"){
                    var firstTop = $($(target + " .routerConnector")[0]).offset().top;
                  } else {
                    var firstTop = $($(target + " .switchConnector")[0]).offset().top;
                  }

                  return firstTop - targetTop;
                }

                // function getConnectorY(n, type){
                //   var t = d3.selectAll(".nodeContainer").filter(function(d){return d.key==n})[0][0];
                //   var connectorY = t.getBoundingClientRect().top + 60;
                //   type == "router" ? connectorY = t.getBoundingClientRect().top + 60 : connectorY = t.getBoundingClientRect().top + 120;
                //   return connectorY;
                // }


                function getRackY(d,i){
                  var rackY = 0;
                   if(i>0){
                     rackY = ( rackHeight * d3.entries(rackArr)[i-1].value ) + rackMargin + tempRackY - 50;
                     tempRackY = rackY;
                   }
                   return rackY;
                }


                function appendText(tTarget, x, y, text, color){
                   tTarget.append("text").attr({
                        "x": x,
                        "y": y
                   }).text(text).style("fill",color);
                }

                function appendRect(rTarget, name, width, height, x, y, color){
                    rTarget.append("rect").attr({
                        "class": "",
                        "width": width,
                        "height": height,
                        "x": x,
                        "y": y 
                    }). style("fill",color);
                }

                function appendImage(iTarget, name, path, width, height, x, y){
                    iTarget.append("image").attr("class", name).attr({
                        "xlink:href" : path,
                        "x" : x,
                        "y" : y,
                        "width" : width,
                        "height" : height,
                    });
                }

};



     var mainBar = {
      sortObj : function(obj, type){
        var arr = [];
        for (var prop in obj) {
            if (obj.hasOwnProperty(prop)) {
                arr.push({ 'key': prop, 'value': obj[prop] });
            }
        }

        if(type == "desc"){ arr.sort(function(a, b) { return b.value - a.value; }); } 
        else if (type == "asc"){ arr.sort(function(a, b) { return a.value - b.value; }); }

        var procObj = {};
        arr.forEach(function(v, i){ procObj[v.key] = v.value; });
        return procObj;
      },
      drawGraph : function(target, data, gWidth, gHeight){
        var margin = {top: 40, right: 20, bottom: 20, left: 80},
            width = gWidth - margin.left - margin.right,
            height = gHeight - margin.top - margin.bottom;


        var color = d3.scale.ordinal().range(["#94CD7A", "#019CCB", "#fff"]);
        var yScale = d3.scale.linear().rangeRound([height, 0]).domain([0, data[0].max]);
        var yAxis = d3.svg.axis().scale(yScale).orient("left").innerTickSize([-30]).tickPadding([10]).ticks(5);

        var svg = d3.select(target).append("svg")
            .attr("width", gWidth)
            .attr("height", gHeight)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

         color.domain(d3.keys(data[0]).filter(function(key) { return key !== "max"; }));
         // color.domain(d3.keys(data[0]));

          data.forEach(function(d) {
            var y0 = 0;
            d.ages = color.domain().map(function(name) { return {name: name, y0: y0, y1: y0 += +d[name]}; });
          });

          svg.append("g").attr("class", "y axis").call(yAxis);

          var state = svg.selectAll(".state")
              .data(data)
              .enter().append("g")
              .attr("class", "g")
              .attr("transform", function(d) { return "translate(0,0)"; });
          
          state.append("rect")
              .attr("class","background")
              .attr("width", width * 0.15)
              .attr("y", 0)
              .attr("height", height)
              .style("fill", "#FFF");

          state.selectAll(".rect")
              .data(function(d) { return d.ages; })
              .enter().append("rect")
              .attr("width", width * 0.15)
              .style("fill", function(d,i) { return color(d.name); })
              .attr("y", function(d) { return height - (yScale(d.y0) - yScale(d.y1)); })
              .transition().delay(function (d,i){ return (i+1) * 800;})
              .duration(800)
              .attr("height", function(d) { return yScale(d.y0) - yScale(d.y1); })
              // .attr("y", function(d) { return yScale(d.y1); })
              // .attr("height", function(d) { return yScale(d.y0) - yScale(d.y1); })
              
      svg.append("defs").append("marker")
        .attr("id", "marker")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 15)
        .attr("refY", -1.5)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
      .append("path")
        .attr("d", "M0,-5L10,0L0,5");


         var ballonContainer = svg.selectAll(".ballonContainer")
              .data(data)
              .enter().append("g")
              .attr("class", "ballonContainer")

         // ballonContainer.selectAll("path")
         //    .data(function(d) { return d.ages; })
         //    .enter().append("path")
         //    .attr("d", "M150 0 L75 200 L225 200 Z")
         //    .attr("x", function(d){ return  width*0.3; })
         //    .attr("y", function(d) { return  height - (yScale(d.y0) - yScale(d.y1)); })
         //    .style("fill", function(d,i) { return color(d.name); });

        ballonContainer.selectAll("rect")
            .data(function(d) { return d.ages; })
            .enter().append("rect")
            .attr("width", 55)
            .attr("x", function(d){ return  width*0.3; })
            .attr("y", function(d) { return  height - (yScale(d.y0) - yScale(d.y1)); })
            .attr("height", function(d) { return 20; })
            .style("fill", function(d,i) { return color(d.name); });

        ballonContainer.selectAll("text")
            .data(function(d) { return d.ages; })
            .enter().append("text")
            .attr("x", function(d){ return  width*0.4; })
            .attr("y", function(d) { return  height - (yScale(d.y0) - yScale(d.y1)) + 15; })
            .text(function(d){ return data[0][d.name] + " GB"; })
            .style("text-anchor","start");

     }
  };

  var mainDonut = {
      target : "#dountChart1",
      getData : function(){
        var that = this;

        var dataset = [
          {label: "Live", count: 80, color: "#94CD7A"},
          {label: "Dead", count: 20, color: "#EF5952"},
        ];

        that.drawGraph(that.target, dataset, 180, 180, "/ 100 Nodes", "Live");
      },
      procData : function(data){
        var that = this;
      },
      drawGraph : function(target, dataset, gWidth, gHeight, sLabel, mLabel){
        var that = this;
          // Dataset mock
        
        // Dimensions of the chart
        var margin = {top: 0, right: 0, bottom: 40, left: 30},
          width = gWidth - margin.left - margin.right,
          height = gHeight - margin.top - margin.bottom;

        var radius = Math.min(width, height)*0.5;

        var svg = d3.select(target).append("svg").attr("width", gWidth).attr("height", gHeight);

        var graphContainer = svg.append("g").attr("class", "graphContainer").attr("transform", "translate(" + (width*0.5 + margin.left) + "," + (height*0.5 + margin.top) + ")");
        var labelContainer = svg.append("g").attr("class", "labelContainer").attr("transform", "translate(" + (width*0.5 + margin.left) + "," + (height*0.5 + margin.top) + ")");
        var legendContainer = svg.append("g").attr("class", "legendContainer").attr("transform", "translate(" + (width*0.25) + "," + (height + margin.bottom - 10) + ")");
        var arc = d3.svg.arc().outerRadius(radius).innerRadius(radius - 15);

        var pie = d3.layout.pie()
            .value(function(d) { return d.count; })
            .sort(null)
            .startAngle(-45 * (Math.PI/180))
            .endAngle(225 * (Math.PI/180))
            .padAngle(0.01);

        graphContainer.selectAll("path")
            .data(pie(dataset))
            .enter()
            .append("path")
            .attr("d", arc)
            .attr("fill", function(d, i) { return d.data.color;})
            .transition()
            .duration(1000)
            .attrTween("d", tweenPie)
            .ease();

          labelContainer.selectAll(".text")
                .data(pie(dataset))
                .enter()
                .append("text")
                .attr("dx","-40")
                .attr("dy", "10")
                .style("text-anchor", "end")
                .style("fill","#fff")
                .style("font-size","25px")
                .text(function(d) { if(d.data.label == mLabel){ return d.value }; });
          
          labelContainer.append("text")
                .attr("dx","-35")
                .attr("dy", "10")
                .style("text-anchor", "start")
                .style("fill","#fff")
                .style("font-size","15px")
                .text(function(d) { return sLabel; });

          var legend = legendContainer.selectAll(".legend")
            .data(pie(dataset))
            .enter().append("g")
            .attr("class", "legend")
            .attr("transform", function(d, i) { return "translate(" + i * 65 + ", -10)"; });

          legend.append("rect")
            .attr("x", 0)
            .attr("width", 10)
            .attr("height", 10)
            .style("fill", function(d){ return d.data.color; });

          legend.append("text")
            .attr("x", 15)
            .attr("y", 7)
            .attr("dy", ".35em")
            .style("text-anchor", "start")
            .style("fill", "#fff")
            .text(function(d) { return d.data.label; });


            function tweenPie(b) {
              var i = d3.interpolate({startAngle: -90 * (Math.PI/180), endAngle: -90 * (Math.PI/180)}, b);
              return function(t) { return arc(i(t)); };
            }

      },
      listener : function(){
        var that = this;
        $(window).resize(function(){

        });
      }
  };