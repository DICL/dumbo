<#ftl encoding="utf-8"/>
<!DOCTYPE html>
<html>



<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Lustre View</title>
    <link rel="shortcut icon" href="${contextPath}/image/favicon.ico">
    
    <script src="${contextPath}/npm/jquery/dist/jquery.js"></script>
	<script src="${contextPath}/npm/axios/dist/axios.js"></script>
	
	
	
	<link rel="stylesheet" href="${contextPath}/npm/font-awesome/css/font-awesome.css" >
	<link rel="stylesheet" href="${contextPath}/npm/ionicons/dist/css/ionicons.css" >
	
	
	
	<script src="${contextPath}/npm/bootstrap/dist/js/bootstrap.js"></script>
	<link rel="stylesheet" href="${contextPath}/npm/bootstrap/dist/css/bootstrap.css" >
	
	<script src="${contextPath}/npm/handlebars/dist/handlebars.js"></script>
	
	<script src="${contextPath}/npm/spin/dist/spin.js"></script>
	
	<script src="${contextPath}/npm/underscore/underscore.js"></script>
	
	<link rel="stylesheet" href="${contextPath}/npm/select2/dist/css/select2.min.css" >
	<script src="${contextPath}/npm/select2/dist/js/select2.min.js"></script>
    
	<link rel="stylesheet" href="${contextPath}/css/index.css" >
	<link rel="stylesheet" href="${contextPath}/css/common.css" >
	<link rel="stylesheet" href="${contextPath}/css/loading.css" >
</head>


<body>
	<div id="root">
		<div class="container">
			<div class="row">
				<!-- loading bar -->
				<div class="pure-loarding">
					<div class="lds-spinner">
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
					</div>
					<p class="message">
						test
					</p>
				</div>
				
				
				
				<input type="hidden" id="fs_num">
				<div class="col-12 mt-5">
					<div class="form-inline">
						<h2>Lustre Setting</h2>
						<button type="button" class="btn btn-primary btn-sm ml-3 operationsRunning" data-target=".bd-managerque-modal-lg">Operations Running</button>
					</div>
					<div id="lustre-setting-container" class="row col-12 border px-0 py-0">
						<div id="fs-list-contailner" class=" px-0 py-0" style="width: 20%">
							<div id="fs-list" class="list-group">
							  <!-- <a href="#" class="list-group-item list-group-item-action active">
							  	lustre1
							  </a>
							  <a href="#" class="list-group-item list-group-item-action">
							  	lustre2
							  </a>
							  <a href="#" class="list-group-item list-group-item-action">
							  	lustre3
							  </a>
							  <a href="#" class="list-group-item list-group-item-action">
								  lustre4
							  </a>
							  <a href="#" class="list-group-item list-group-item-action">
								  lustre5
							  </a> -->
							</div>
							<div id="fs-buttons">
								<button id="fs-add-btn" type="button" class="btn btn-primary">Add</button>
								<button id="fs-del-btn" type="button" class="btn btn-primary">Delete</button>
							</div>
						</div>
						<div id="fs-view" class="px-0 py-0" style="width: 80%">
							<div id="fs-menu" class="col-12 px-3 py-3" style="display: none;">
								<ul class="nav nav-pills nav-fill">
									<li class="nav-item"><a id="MDS" class="nav-link active" href="${contextPath}/">MDS Setting</a></li>
									<li class="nav-item"><a id="OSS_Setting" class="nav-link" href="${contextPath}/OSS_Setting">OSS Setting</a></li>
									<li class="nav-item"><a id="Client_Setting" class="nav-link" href="${contextPath}/Client_Setting">Client Setting</a></li>
									<li class="nav-item"><a id="LNET_Setting" class="nav-link" href="${contextPath}/LNET_Setting">LNET Setting</a></li>
									<li class="nav-item"><a id="Backup" class="nav-link" href="${contextPath}/Backup">Backup</a></li>
									<li class="nav-item"><a id="Restore" class="nav-link" href="${contextPath}/Restore">Restore</a></li>
								</ul>
							</div>
							<!-- #fs-view-container start -->
							<div id="fs-view-container" class="col-12">
								<!-- <div class="col-12 px-3 py-3">
									<div class="form-group col-12">
								      <label for="mdt">
								      	<span class="mds-hostname">
								      		master.hadoop.com
								      	</span>
								      	 MDT device and size
								      </label>
								      <div class="form-inline">
								      	<select id="mds-disk-name" class="form-control col-6">
									      	
									      </select>
									      <small id="mds-disk-info" class="ml-5">
									      	
									      </small>
								      </div>
								    </div>
								</div>
								
								<div class="col-12 px-3 py-3">
									<div class="form-group col-12">
								      <label for="mdt">
								       <span class="mds-hostname">
								      		master.hadoop.com
								      	</span>
								       IO network
								       </label>
								      <div class="form-inline">
								      	<select id="mds-network-info" class="form-control col-6">
									      	
									      </select>
									      <small id="" class="ml-5">
									      	
									      </small>
								      </div>
								      
								    </div>
								</div>
								<div class="col-12 px-3 py-3 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Apply</button>
									<button type="button" id="mds-reset" class="btn btn-primary">Reset</button>
								</div> -->
							</div>
							<!-- #fs-view-container end -->
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
	
	<script>
	var contextPath = "${contextPath}";
	</script>
	
	<script type="text/javascript" src="${contextPath}/js/common.js"></script>
	<script type="text/javascript" src="${contextPath}/js/Handlebars.custom.js"></script>
	
	<script type="text/javascript" src="${contextPath}/js/operationsRunning.js"></script>
	<script type="text/javascript" src="${contextPath}/js/page/index.js"></script>
	<script type="text/javascript" src="${contextPath}/js/fs_request.js"></script>
	
	<!-- bootstrap-switch -->
	<link rel="stylesheet" href="${contextPath}/css/bootstrap-switch.css" >
	<script type="text/javascript" src="${contextPath}/js/bootstrap-switch.js"></script>
	
	<!-- bootstrap-toggle -->
	<link rel="stylesheet" href="${contextPath}/js/bootstrap-toggle/bootstrap-toggle.css" />
	<script type="text/javascript" src="${contextPath}/js/bootstrap-toggle/bootstrap-toggle.js"></script>
	
	<div class="modal fade bd-managerque-modal-lg" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true">
	  <div class="modal-dialog modal-lg">
	    <div class="modal-content">
	      <div class="modal-header">
	        <h5 class="modal-title">Job List</h5>
	        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
	          <span aria-hidden="true">&times;</span>
	        </button>
	      </div>
	      <div class="modal-body">
	        
	      </div>
	      <div class="modal-footer">
	        <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
	      </div>
	    </div>
	  </div>
	</div>
	
	<div class="modal fade bd-file_system_step-modal-lg" tabindex="-1" role="dialog" aria-labelledby="file_system_step" aria-hidden="true" data-backdrop="static" data-keyboard="false">
	  <div class="modal-dialog modal-lg">
	    <div class="modal-content">
	      <div class="modal-header">
	        <h5 class="modal-title">Job List</h5>
	        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
	          <span aria-hidden="true">&times;</span>
	        </button>
	      </div>
	      <div class="modal-body">
	        
	      </div>
	      <div class="modal-footer">
	        <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
	      </div>
	    </div>
	  </div>
	</div>
</body>


</html>