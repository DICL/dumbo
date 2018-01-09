// var superCom = angular.module('superCom',['ngRoute'])
// superCom.config(function($routeProvider,$locationProvider){
angular.module('superCom',['ngRoute'])
.config(function($routeProvider){
	
	$routeProvider
	  .when("/",{ 
	  	templateUrl:"template/main.html",
	  	activetab : 'main',
	  	controller : 'mainController'
	  })
	  .when("/main",{ 
	  	templateUrl:"template/main.html",
	  	activetab : 'main',
	  	controller : 'mainController'
	  })
	  .when("/lustre",{ 
	  	templateUrl:"template/lustreFS.html",
	  	activetab : 'lustre',
	  	controller : 'lustreController'
	  })
	  .when("/yarn",{ 
	  	templateUrl:"template/yarn.html",
	  	activetab : 'yarn',
	  	controller : "YarnDataController"
	  })
	  .when("/ambari",{ 
	  	templateUrl:"template/main.html",
	  	activetab : 'main', 
	  	controller : "ambariController"
	  })
	  .otherwise({
	  	redirectTo:'/main', 
	  	templateUrl:"template/main.html",
	  	activetab:"main",
	  	controller : 'mainController'
	  });
	  
})
.run(function($rootScope,$route,$sce){
	$rootScope.$route = $route;
	// leftGraphInit($rootScope);
	
	var base_url = 'http://192.168.1.194:8080/dashboard-solo/db/';
	$rootScope.graph_url = {
		// main,lustre => graph lustre url
		lustre_1 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=1&fullscreen&theme=light'),
		lustre_2 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=2&fullscreen&theme=light'),
		lustre_3 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=3&fullscreen&theme=light'),
		lustre_4 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=4&fullscreen&theme=light'),
		lustre_5 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=5&fullscreen&theme=light'),
		lustre_6 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=6&fullscreen&theme=light'),
		lustre_7 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=7&fullscreen&theme=light'),
		lustre_8 : $sce.trustAsResourceUrl(base_url+'lustrefs?panelId=8&fullscreen&theme=light'),

		// main,yarn => graph yarn url
		yarn_3 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=3&fullscreen&theme=light'),
		yarn_10 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=10&fullscreen&theme=light'),
		yarn_6 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=6&fullscreen&theme=light'),
		yarn_12 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=12&fullscreen&theme=light'),
		yarn_11 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=11&fullscreen&theme=light'),
		yarn_14 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=14&fullscreen&theme=light'),
		yarn_13 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=13&fullscreen&theme=light'),
		yarn_15 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=15&fullscreen&theme=light'),
		yarn_16 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=16&fullscreen&theme=light'),
		yarn_17 : $sce.trustAsResourceUrl(base_url+'yarn-home2?panelId=17&fullscreen&theme=light'),

		// main => graph gpu url
		gpu_1 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=1&fullscreen&theme=light'),
		gpu_2 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=2&fullscreen&theme=light'),
		gpu_3 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=3&fullscreen&theme=light'),
		gpu_4 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=4&fullscreen&theme=light'),
		gpu_5 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=5&fullscreen&theme=light'),
		gpu_6 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=6&fullscreen&theme=light'),
		gpu_7 : $sce.trustAsResourceUrl(base_url+'gpu-monitor?panelId=7&fullscreen&theme=light')
	}	

})
.controller('ambariController',function($scope){
	$("#loading").addClass("hide");
	window.open('http://192.168.1.194:8080/','_blank');
})
.controller('mainController',function($scope,$http,$q){
	console.log("mainController start");
	// $.ajaxSetup({
 //        xhrFields: { withCredentials: true },
 //        beforeSend: function(xhr) {  xhr.setRequestHeader("Authorization", "Basic " + btoa("admin:admin")); }
 //    });
	// clusterInfo.getClusterName($scope,$http);
	clusterInfo($scope,$http,$q);
	$scope.getClusterName();
    $("#chameleonNode").removeClass("hide");
    $("#loading").addClass("hide");
	listener();
	$scope.nodePopup = function(){
		console.log("test::click mainController");
	}
})
// yarn controller(yarn angularjs)
.controller('YarnDataController',function($scope,$http){
    $("#loading").addClass("hide");
	// scope get Data
	$http({
	    method: 'GET',
	    url: 'http://192.168.1.195:8088/ws/v1/cluster/apps',
	    headers: {'Content-Type': 'application/json; charset=utf-8'}
    })
    .success(function(data){
    	console.log("success data yarn::");
        // console.log(data.apps.app);
		$scope.items = data.apps.app;
	    $scope.groupToPages();
    })
    .error(function(err){
    	console.log("YarnDataController error");
    	console.log(err);
    });

    // history click event
	$scope.click = function(itemId) {
		
    	var applicationId = (itemId);
    	var jobId = "job" + applicationId.substring(11, applicationId.length);
        $http({
	        method: 'GET',
	        //url: 'http://192.168.1.195:19888/ws/v1/history/mapreduce/jobs/{job_id}',
	        url: 'http://192.168.1.195:19888/ws/v1/history/mapreduce/jobs/'+jobId,
	        headers: {'Content-Type': 'application/json; charset=utf-8'}
        }).success(function(data){
            console.log(data);
            $scope.items = data;
            // alert(data);
	        //$scope.items = data;
        }).error(function(){
            alert('History가 존재 하지 않습니다.');  
        });
    };	

    // pagination
    $scope.gap = 5;
	$scope.groupItems = [];
	$scope.itemsPerPage = 5;
	$scope.pagedItems = [];
	$scope.currentPage = 0;
	$scope.groupToPages = function(){
		$scope.pagedItems = [];
		for(var i=0; i<$scope.items.length; i++){
			if(i % $scope.itemsPerPage === 0){
				$scope.pagedItems[Math.floor(i/$scope.itemsPerPage)] = [$scope.items[i]];
			}else{
				$scope.pagedItems[Math.floor(i/$scope.itemsPerPage)].push($scope.items[i]);
			}
		}
	};
	$scope.range = function(size,start,end){
		var ret = [];
		if(size < end){
			end = size;
			if(size<$scope.gap){
				start = 0;
			}else{
				start = size-$scope.gap;
			}
		}
		for(var i=start; i<end; i++){
			ret.push(i);
		}
		return ret;
	};
	$scope.prevPage = function(){
		if($scope.currentPage > 0){
			$scope.currentPage--;
		}
	};
	$scope.nextPage = function(){
		if($scope.currentPage < $scope.pagedItems.length-1){
			$scope.currentPage++;
		}
	};
	$scope.setPage = function(){
		$scope.currentPage = this.n;
	};
    
})
.controller('lustreController',function($scope){
	console.log('::lustreController::');
	$("#loading").addClass("hide");
	$scope.popup = function(){
		alert("test");
	}	
});

var clusterInfo = function($scope,$http,$q){
	
	$scope.getClusterName = function(){
		var url = location.origin + "/api/v1/clusters";
		var username = "admin", password = "admin";
		var config = {
			headers: {
			   Authorization: "Basic " + window.btoa(username+":"+password)
			},
			method: "get",
			url: url
		};
		$http(config)
		.success(function(res){
			// console.log("SUCCESS::",res);
			$scope.clusterNameitmes = res.items;
			$scope.clusterName = $scope.clusterNameitmes[0].Clusters.cluster_name
			$scope.getClusterDetail($scope,$http);
		}).error(function(err){
			console.log("ERROR");
			console.log(err);
		});

		// var data = {user:"admin",pass:"admin"};
		// $http.post(url,data)
		// .then(function success(res){
		// 	// console.log("SUCCESS::",res);
		// 	$scope.clusterNameitmes = res.data.items;
		// 	$scope.clusterName = $scope.clusterNameitmes[0].Clusters.cluster_name
		// 	$scope.getClusterDetail($scope,$http);
		// },
		// function error(err){
		// 	console.log("ERROR");
		// 	console.log(err);
		// });
	};

	$scope.getClusterDetail = function(){
		$http({
			url : location.origin + "/api/v1/clusters/" + $scope.clusterName + "/services/HDFS/components/NAMENODE"
		})
		.success(function(data){
			$scope.clusterDetailItems = data;
			var CapacityUsed = $scope.clusterDetailItems.ServiceComponentInfo.CapacityUsed;
			if(!CapacityUsed){
				CapacityUsed = 0;
			}
			var CapacityTotal = $scope.clusterDetailItems.ServiceComponentInfo.CapacityTotal;
			if(!CapacityTotal){
				CapacityTotal = 0;
			}
			var dfsUsed =  (( CapacityUsed / CapacityTotal ) * 100).toFixed(2);
			if(!dfsUsed){
				dfsUsed = 0;
			}
			var NonDfsUsedSpace = $scope.clusterDetailItems.ServiceComponentInfo.NonDfsUsedSpace;
			if(!NonDfsUsedSpace){
				NonDfsUsedSpace = 0;
			}
			var PercentUsed = $scope.clusterDetailItems.ServiceComponentInfo.PercentUsed;
			if(!PercentUsed){
				PercentUsed = 0;
			}
			var CapacityRemaining = $scope.clusterDetailItems.ServiceComponentInfo.CapacityRemaining;
			if(!CapacityRemaining){
				CapacityRemaining = 0;
			}
			var PercentRemaining = $scope.clusterDetailItems.ServiceComponentInfo.PercentRemaining;
			if(!PercentRemaining){
				PercentRemaining = 0;
			}
			var html = "";
			html += " - DFS Used : "+ (CapacityUsed/ Math.pow(1024, 4)).toFixed(2) + " / " + (CapacityTotal/ Math.pow(1024, 4)).toFixed(2) +" TB ("+dfsUsed+" %) <br/>";
			html += " - Non DFS Used : "+ (NonDfsUsedSpace/ Math.pow(1024, 4)).toFixed(2) + " / " + (CapacityTotal/ Math.pow(1024, 4)).toFixed(2) +" TB ("+PercentUsed.toFixed(2)+" %) <br/>";
			html += " - DFS Remaining : "+ (CapacityRemaining/ Math.pow(1024, 4)).toFixed(2) + " / " + (CapacityTotal/ Math.pow(1024, 4)).toFixed(2) +" TB ("+PercentRemaining.toFixed(2)+" %) <br/>";
			$(".cluster_wrap[data-index=1] .cluster_info .cluster_detail_desc").html(html);
			$scope.getNodeList();

		})
		.error(function(err){
			console.log("error");
		    console.log(err);
		})
	};

	$scope.getNodeList = function(){
		$http({
			url : location.origin + "/api/v1/clusters/" + $scope.clusterName + "/hosts"
		})
		.success(function(data){
			$scope.nodeItems = data;
			$scope.hostArr = [];
			for(var i=0; i<$scope.nodeItems.items.length; i++){
				$scope.hostArr.push($scope.nodeItems.items[i].Hosts.host_name);
			}
			$scope.getNodeDetail();
			// $scope.getNodeDetail().then(function(){
			// 	console.log("::test::then:::");
			// 	$scope.drawNode();
			// });
			// $q.all([$scope.getNodeDetail()]).then(function(){
			// 	console.log("TEST:::Q:::");
			// 	$scope.drawNode();
			// })
			// that.getNodeDetail(hostArr);
		})
		.error(function(err){
			console.log("error");
			console.log(err);
		})
	};
	$scope.getNodeDetail = function(){
		
		$scope.nodeDataArr = [];
		$scope.lusterArr = [];
		var endTime = new Date().getTime();
		var startTime = new Date(endTime - 3600000).getTime();

		angular.forEach($scope.hostArr, function(d,i){
			getNodeDataArr(i);
			getLusterArr(i);
			
		});

		function getNodeDataArr(i){
			$http({
				url : location.origin + "/api/v1/clusters/"+$scope.clusterName+"/hosts/" + $scope.hostArr[i] + "?fields=metrics/cpu,metrics/disk,metrics/memory,metrics/network"
			})
			.success(function(data){
				$scope.nodeDataArr.push(data);
				if($scope.hostArr.length == $scope.nodeDataArr.length){
					$scope.drawNode();		
				}	
			})
			.error(function(err){
				console.log("error");
				console.log(err);
				
			});
			nodeListener();
		}

		function getLusterArr(i){
			$http({
				url: location.origin + "/api/v1/clusters/"+$scope.clusterName+"/hosts/" + $scope.hostArr[i] + "/host_components"
			})
			.success(function(data){
				var ajaxData = data;
				ajaxData.items.forEach(function(d){
					if(d.HostRoles.component_name == "LUSTRE"){
						$scope.lusterArr.push(d.HostRoles.host_name.split(".")[0]);
					}
				})
			})
			.error(function(err){
				console.log("error");
				console.log(err);
			})
		}
	}
	// $scope.$watch('nodeDataArr',function(){
	// 	$scope.drawNode();
	// },true);

	$scope.drawNode = function(){
		var html = '';
		
		var tempData = $scope.nodeDataArr;
		if(tempData){
			drawTempNode(tempData);
		}
		function drawTempNode(tempData){
			tempData.forEach(function(data,i){

				var nodeName = data.Hosts.host_name.split(".")[0];
				if(!nodeName){
					nodeName = 0;
				}
				if(data.metrics){
					var cpu = data.metrics.cpu.cpu_user;
					if(!cpu){
						cpu = 0;
					}
					var ram = ((data.metrics.memory.mem_total - data.metrics.memory.mem_free) / data.metrics.memory.mem_total * 100).toFixed(1);
					if(!ram){
						ram = 0;
					}
					var disk = ((data.metrics.disk.disk_total - data.metrics.disk.disk_free) / data.metrics.disk.disk_total * 100).toFixed(1);
					if(!disk){
						disk = 0;
					}
					var networkIn = (data.metrics.network.bytes_in / Math.pow(1024, 1)).toFixed(1);
					if(!networkIn){
						networkIn = 0;
					}
					var networkOut = (data.metrics.network.bytes_out / Math.pow(1024, 1)).toFixed(1);
					if(!networkOut){
						networkOut = 0;
					}
				}else{
					console.log("data.metrics is empty");
					return;
				}
				var lusterYn = $scope.lusterArr.indexOf(nodeName) > -1 ? "Y" : "N";
				html+= '  <div class="cluster_node pull-left"> ';
				html+= '      <div class="node_name" data-name="'+nodeName+'" ng-click="nodePopup()">'+nodeName+'</div> ';
				html+= '      <div class="node_status" data-luster="'+lusterYn+'"> ';
				html+= '          <div class="status" data-status="'+$scope.getNodeStatus(cpu)+'" data-type="cpu">cpu '+cpu+'%</div> ';
				html+= '          <div class="status" data-status="'+$scope.getNodeStatus(ram)+'" data-type="ram">ram '+ram+'%</div> ';
				html+= '          <div class="status" data-status="'+$scope.getNodeStatus(disk)+'" data-type="disk">disk '+disk+'%</div> ';
				html+= '          <div class="status" data-status="default" data-type="ni">ni '+networkIn+'KB/s</div> ';
				html+= '          <div class="status" data-status="default" data-type="no">no '+networkOut+'KB/s</div> ';
				html+= '      </div> ';
				html+= '  </div> ';
				if(i==$scope.nodeDataArr.length-1){ html += ' <div class="clearfix"></div>'; }
			});	
		}

		$(".cluster_node_wrap").html(html);

		leftGraphInit($scope.clusterName);
	}

	function nodeListener(){
		
		var target = ".right_wrap .contents_wrap .cluster_wrap .cluster_node .node_status";
		var cHeight = $(target).height();
		var len = $(".cluster_btn_wrap .cluster_btn.active").length;
		$(target + " .status").css("line-height", cHeight/len + "px");

		$(".cluster_btn_wrap .cluster_btn").off("click").on("click",function(e){
 			e.preventDefault();
			// console.log("cluster_btn click event::");
			// var checked = $(".cluster_btn_wrap .cluster_btn.active").length;
			// console.log("::active checked::",checked);
			
			// if($(this).hasClass("active") == false){
			// 	console.log("active false");
			// 	$(this).addClass("active");
			// }else{
			// 	console.log("active true");
			// 	$(this).removeClass("active");
			// }
			// if(checked == 0){ $(this).addClass("active"); }
			// else { $(this).toggleClass("active"); }
			$(this).toggleClass("active");

			var type = $(this).attr("menu");

			if($(this).hasClass("active")){
				$(target + " .status[data-type="+type+"]").removeClass("hide");
			} else {
				$(target + " .status[data-type="+type+"]").addClass("hide");
			}

			var cHeight = $(target).height();
			var len = $(".cluster_btn_wrap .cluster_btn.active").length;
			$(target + " .status").css("line-height", cHeight/len + "px");
		});





		$(".cluster_btn_wrap .luster_btn").off("click").on("click",function(){
			$(this).toggleClass("active");
			var target = ".right_wrap .contents_wrap .cluster_wrap .cluster_node .node_status";

			if($(this).hasClass("active")){
			  $(target+"[data-luster=Y]").css("box-shadow","2px 2px 4px rgba(0,0,0,0.6)");
			} else {
			  $(target+"[data-luster=Y]").css("box-shadow","none");
			}
		});
	}

	$scope.getNodeStatus = function(d){
		if(d>=0 && d<=20){ return 1; }
		else if(d>=21 && d<=40){ return 2; }
		else if(d>=41 && d<=60){ return 3; }
		else if(d>=61 && d<=80){ return 4; }
		else if(d>=81 && d<=100){ return 5; }
		else { return "default"; }
	}

}


function leftGraphInit(clusterName){
      $.when(leftGraph.getLiveStatus(clusterName), leftGraph.getStorageUsage(clusterName), leftGraph.getResourceHeap(clusterName), leftGraph.getYarnMemory(clusterName), leftGraph.getNamenodeHeap(clusterName), leftGraph.getAlert(clusterName)).done(function (result1, result2, result3, result4, result5, result6) {

          // left Graph
          if(result1.ServiceComponentInfo){
            leftGraph.drawLiveStatus(result1);  
          }
          if(result2.ServiceComponentInfo){
            leftGraph.drawStorageUsage(result2);  
          }
          if(result3.metrics){
            leftGraph.drawResourceHeap(result3);  
          }
          if(result4.metrics){
            leftGraph.drawYarnMemory(result4);  
          }
          if(result5.ServiceComponentInfo){
            leftGraph.drawNamenodeHeap(result5);  
          }
          if(result6){
            leftGraph.drawAlert(result6,clusterName);  
          }

          // loading hide
          $("#chameleonNode").removeClass("hide");
          $("#loading").addClass("hide");

      });
}

var leftGraph = {
	drawLiveStatus: function(ajaxData){
	  var started_count = ajaxData.ServiceComponentInfo.started_count;
	  var total_count = ajaxData.ServiceComponentInfo.total_count;

	  var graph_target = "#liveStatusBar";
	  $(graph_target + " .graph_label .value").html(started_count);
	  $(graph_target + " .graph_label .total").html(" / "+total_count+" Nodes");

	  var valuePercent = (( started_count / total_count ) * 100 ).toFixed(0);
	  $(graph_target + " .bar_graph .value1").css("width", valuePercent+"%");
	  $(graph_target + " .bar_graph .value2").css("width", 100-valuePercent+"%");


	},

	drawStorageUsage : function(ajaxData){
	  var CapacityUsed = (ajaxData.ServiceComponentInfo.CapacityUsed / Math.pow(1024, 4)).toFixed(0);
	  var CapacityRemaining = (ajaxData.ServiceComponentInfo.CapacityRemaining  / Math.pow(1024, 4)).toFixed(2);
	  var CapacityTotal = (ajaxData.ServiceComponentInfo.CapacityTotal / Math.pow(1024, 4)).toFixed(2);

	  
	  var graph_target = "#storageUsageBar";
	  $(graph_target + " .graph_label .value").html(CapacityUsed);
	  $(graph_target + " .graph_label .total").html(" / "+CapacityTotal+" TB");

	  var valuePercent1 = (( CapacityUsed / CapacityTotal ) * 100 ).toFixed(0);
	  var valuePercent2 = (( CapacityRemaining / CapacityTotal ) * 100 ).toFixed(0);
	  $(graph_target + " .bar_graph .value1").css("width", valuePercent1+"%");
	  $(graph_target + " .bar_graph .value2").css("width", valuePercent2+"%");

	},

	drawResourceHeap : function(ajaxData){
	  var HeapMemoryMax = (ajaxData.metrics.jvm.HeapMemoryMax / Math.pow(1024, 2)).toFixed(2);
	  var HeapMemoryUsed = (ajaxData.metrics.jvm.HeapMemoryUsed  / Math.pow(1024, 2)).toFixed(2);

	  var graph_target = "#resourceHeapBar";
	  $(graph_target + " .graph_label .value").html(HeapMemoryUsed);
	  $(graph_target + " .graph_label .total").html(" / "+HeapMemoryMax+" MB");

	  var valuePercent = (( HeapMemoryUsed / HeapMemoryMax ) * 100 ).toFixed(0);
	  $(graph_target + " .bar_graph .value1").css("width", valuePercent+"%");
	  $(graph_target + " .bar_graph .value2").css("width", 100-valuePercent+"%");

	},

	drawYarnMemory : function(ajaxData){

	  var AvailableMB = (ajaxData.metrics.yarn.Queue.root.AvailableMB / Math.pow(1024, 1)).toFixed(2) * 1;
	  var AllocatedMB = (ajaxData.metrics.yarn.Queue.root.AllocatedMB  / Math.pow(1024, 1)).toFixed(2) * 1;
	  var Total = AvailableMB + AllocatedMB;

	  var graph_target = "#yarnMemoryBar";
	  $(graph_target + " .graph_label .value").html(AllocatedMB);
	  $(graph_target + " .graph_label .total").html(" / "+Total+" GB");

	  var valuePercent = (( AllocatedMB / Total ) * 100 ).toFixed(0);
	  $(graph_target + " .bar_graph .value1").css("width", valuePercent+"%");
	  $(graph_target + " .bar_graph .value2").css("width", 100-valuePercent+"%");
	},

	drawNamenodeHeap : function(ajaxData){
	  var HeapMemoryMax = (ajaxData.ServiceComponentInfo.HeapMemoryMax / Math.pow(1024, 2)).toFixed(2) * 1;
	  var HeapMemoryUsed = (ajaxData.ServiceComponentInfo.HeapMemoryUsed  / Math.pow(1024, 2)).toFixed(2) * 1;
	  
	  var graph_target = "#namenodeHeapBar";
	  $(graph_target + " .graph_label .value").html(HeapMemoryUsed);
	  $(graph_target + " .graph_label .total").html(" / "+HeapMemoryMax+" MB");

	  var valuePercent = (( HeapMemoryUsed / HeapMemoryMax ) * 100 ).toFixed(0);
	  $(graph_target + " .bar_graph .value1").css("width", valuePercent+"%");
	  $(graph_target + " .bar_graph .value2").css("width", 100-valuePercent+"%");

	},

	drawAlert : function(ajaxData,cName){
	var that = this;
	  for(var i=0; i<3; i++){
	      var aId = ajaxData.items[i].Alert.id;
	      that.getAlertDetail(aId,cName);
	  }

	},
	getLiveStatus : function(cName){
		var deferred = $.Deferred();
		 $.ajax({
		    url: location.origin + "/api/v1/clusters/"+cName+"/services/HDFS/components/DATANODE?fields=ServiceComponentInfo",
		    success : function(data){
		        // console.log("::::::::::::::::getLiveStatus:::::::::::::::: \n", data);

		        var ajaxData = JSON.parse(data);
		        deferred.resolve(ajaxData);
		    },
		    error : function(err){
		        console.log("error");
		    	console.log(err);
		    }
		 });
		return deferred;
	},

	getStorageUsage : function(cName){
	var deferred = $.Deferred();
	 $.ajax({
	    url: location.origin + "/api/v1/clusters/"+cName+"/services/HDFS/components/NAMENODE?fields=ServiceComponentInfo",
	    success : function(data){
	        // console.log("::::::::::::::::getStorageUsage:::::::::::::::: \n", data);
	        var ajaxData = JSON.parse(data);
	        deferred.resolve(ajaxData);
	    },
	    error : function(err){
	      console.log("error");
	       console.log(err);
	    }
	 });
	return deferred;
	},

	getResourceHeap : function(cName){
	var deferred = $.Deferred();
	 $.ajax({
	  url: location.origin + "/api/v1/clusters/"+cName+"/services/YARN/components/RESOURCEMANAGER?fields=metrics/jvm",
	    success : function(data){
	        // console.log("::::::::::::::::getResourceHeap:::::::::::::::: \n", data);

	        var ajaxData = JSON.parse(data);
	         deferred.resolve(ajaxData);
	    },
	    error : function(err){
	      console.log("error");
	       console.log(err);
	    }
	 });
	 return deferred;
	},

	getYarnMemory : function(cName){
	var deferred = $.Deferred();
	 $.ajax({
	    url: location.origin + "/api/v1/clusters/"+cName+"/services/YARN/components/RESOURCEMANAGER?fields=metrics/yarn/Queue/root",
	    success : function(data){
	        // console.log("::::::::::::::::getYarnMemory:::::::::::::::: \n", data);

	        var ajaxData = JSON.parse(data);
	         deferred.resolve(ajaxData);
	    },
	    error : function(err){
	      console.log("error");
	       console.log(err);
	    }
	 });
	 return deferred;
	},

	getNamenodeHeap : function(cName){
	var deferred = $.Deferred();
	 $.ajax({
	    url: location.origin + "/api/v1/clusters/"+cName+"/services/HDFS/components/NAMENODE?fields=ServiceComponentInfo",
	    success : function(data){
	        // console.log("::::::::::::::::getNamenodeHeap:::::::::::::::: \n", data);

	        var ajaxData = JSON.parse(data);
	         deferred.resolve(ajaxData);
	    },
	    error : function(err){
	      console.log("error");
	       console.log(err);
	    }
	 });
	 return deferred;
	},

	getAlert : function(cName){
	 var that = this;
	 var deferred = $.Deferred();
	 $.ajax({
	    url: location.origin + "/api/v1/clusters/"+cName+"/alerts",
	    success : function(data){
	        // console.log("::::::::::::::::getAlert:::::::::::::::: \n", data);

	        var ajaxData = JSON.parse(data);
	        deferred.resolve(ajaxData);
	    },
	    error : function(err){
	      console.log("error");
	       console.log(err);
	    }
	 });
	  return deferred;
	},

	getAlertDetail : function(aId,cName){
	 var that = this;
	 $.ajax({
	    url: location.origin + "/api/v1/clusters/"+cName+"/alerts/"+aId,
	    success : function(data){
	        // console.log("::::::::::::::::getAlertDetail:::::::::::::::: \n", data);
	        var ajaxData = JSON.parse(data);
	        that.drawAlertDetail(ajaxData.Alert);
	    },
	    error : function(err){
	      console.log("error");
	       console.log(err);
	    }
	 });
	},

	drawAlertDetail : function(data){
	var time = new Date(data.latest_timestamp).format('MM-dd hh:mm'); 
	var alert_name = data.definition_name.replace(/_/gi, " ");;

	var target = ".alerts_popover .pop_desc";
	var html = '';
	// html += alert_name + " : " + data.text + " : " + time;
	html += ' <div class="con"> ';
	html += '    <div class="con_title"><i class="fa fa-circle"></i>'+alert_name+'</div> <div class="clearfix"></div> ';
	html += '    <div class="con_desc">'+data.text+'</div> <div class="clearfix"></div> ';
	html += '    <div class="con_val">'+time+'</div> ';
	html += ' </div> ';

	$(target).append(html);
	}

};

// listener
function listener(){
	$(".left_wrap_btn").off("click").on("click",function(){
	    var that = this;
	    var target = $(that).siblings(".left_wrap");

	    if($(target).hasClass("nav_close")){
	      $(target).animate({ width: "21%" }, 500).removeClass("nav_close");
	      $(".right_wrap").animate({ width: "75%" }, 500);
	      line1.resize({ width: 370 });
	      line2.resize({ width: 370 });
	      line3.resize({ width: 370 });
	      line4.resize({ width: 370 });
	      line4.resize({ width: 370 });
	    } else {
	      $(target).animate({ width: "20px" }, 500).addClass("nav_close");
	      $(".right_wrap").animate({ width: "95%" }, 500);
	      line1.resize({ width: 480 });
	      line2.resize({ width: 480 });
	      line3.resize({ width: 480 });
	      line4.resize({ width: 480 });
	    }
	});

	$("#nav .nav_icon.alerts img").on("mouseover",function(e){
	    $("#nav .alerts_popover").removeClass("hide").css({"left": (e.pageX-350) + "px", "top": (e.pageY-40)+"px"});; 
	}).on("mouseout",function(){ 
	  $("#nav .alerts_popover").addClass("hide"); 
	});

}