<#ftl encoding="utf-8"/>


<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Metric Registry View</title>
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
			  	
			  	<!-- menu -->
				<div class="col-12">
					<div class="form-inline">
						<h2>Metric Registry View</h2>
					</div>
				</div>
			  	
				<div class="col-12 mt-1">
					<div class="col-12">
						<h5>Registered metrics:</h5>
						<div class="metric-cycle-time form-inline col-12 px-3 py-3">
							<label>Crontab Cycle : </label>
							<input name="cycle_time" id="cycle_time" type="text" class="ml-3 form-control" value=""/>
							<label class="ml-3">Second</label>
							<button type="button" id="update_cycle_time" class="ml-3 btn btn-primary">Update</button>
						</div>
						<div id="metric_list" class="col-12 px-3 py-3 border">
							<!-- <div class="row mb-3">
								<div class="col-6">
									pidstat.cpu
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									pidstat.mem
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									pidstat.diskio.read
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									pidstat.diskio.write
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-6">
									perf.branch-misses
								</div>
								<div class="col-6 text-right">
									<button type="button" id="mds-apply" class="btn btn-primary">Del</button>
									<button type="button" id="mds-reset" class="btn btn-primary">View</button>
								</div> -->
							</div>
						</div>
						<div class="col-12 px-3 py-3 text-right">
							<!-- <button type="button" id="" class="btn btn-primary">Add</button> -->
							<a href="${contextPath}/addMetric" class="btn btn-primary">Add</a>
						</div>
					</div>
					
				</div>
			</div>
		</div>
</body>

<script>
var contextPath = "${contextPath}";
</script>

<script type="text/javascript" src="${contextPath}/js/common.js"></script>
<script type="text/javascript" src="${contextPath}/js/Handlebars.custom.js"></script>
<script type="text/javascript" src="${contextPath}/js/operationsRunning.js"></script>
<script type="text/javascript" src="${contextPath}/js/page/Metric_Registry_View.js"></script>

<div class="modal fade bd-managerque-modal-lg" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true" data-backdrop="static" data-keyboard="false">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Init Table</h5>
        <button type="button" id="close-button" class="close" data-dismiss="modal" aria-label="Close" style="display: none;">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body" id="is_message">
        Table not created.
		Do you want to create a table?
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" id="create-table">Create Table</button>
      </div>
    </div>
  </div>
</div>
</html>