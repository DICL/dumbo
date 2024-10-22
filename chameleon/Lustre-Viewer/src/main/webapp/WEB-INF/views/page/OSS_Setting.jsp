<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html>

<jsp:include page="commons/head.jsp" flush="true"></jsp:include>

<link rel="stylesheet" href="${contextPath}/css/bootstrap-switch.css" >
<script type="text/javascript" src="${contextPath}/js/bootstrap-switch.js"></script>

<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<jsp:include page="commons/header.jsp" flush="true"></jsp:include>
			  	
			  	<div class="col-12 mt-5">
					<h2>OSS Setting</h2>
					<div id="oss_page" class="col-12 border px-3 py-3">
						<ul class="nav nav-tabs" id="myTab" role="tablist">
						  <li class="nav-item">
						    <a class="nav-link active" id="home-tab" data-toggle="tab" href="#home" role="tab" aria-controls="home" aria-selected="true">
						    	OSS1 Setting
						    </a>
						  </li>
						  <li class="nav-item">
						    <a class="nav-link" id="profile-tab" data-toggle="tab" href="#profile" role="tab" aria-controls="profile" aria-selected="false">
						    	OSS2 Setting
						    </a>
						  </li>
						</ul>
						<div class="tab-content" id="myTabContent">
						  <div class="tab-pane fade show active" id="home" role="tabpanel" aria-labelledby="home-tab">
						  	<div class="col-12 border-left border-right  px-3 py-3">
							  	<h5>slave1.hadoop.com OST device and size</h5>
							  	<div class="form-inline mt-3">
							  		<div class="form-group col-3 px-0">
								  		<select id="mds-disk-name" class="form-control col-6">
									  		<option>OST1</option>
									  		<option>OST2</option>
									  		<option>OST3</option>
									  		<option>OST4</option>
								      	</select>
							  		</div>
							  		
							  		<div class="form-group col-3 px-0">
								  		<select id="" class="form-control">
							  				<option>/dev/sdb</option>
							      		</select>
							  		</div>
							  		
							      	<div class="form-group col-3 px-0">
								  		<small>100GB</small>
							  		</div>
							      	
							      	<div class="col-3 text-right">
							      		<button type="button" class="btn btn-primary">activate</button>
							      		<button type="button" class="btn btn-primary">deactivate</button>
							      	</div>
							      	
							  	</div>
							  	<div class="form-inline mt-3">
							  		<div class="form-group col-3 px-0">
								  		<select id="mds-disk-name" class="form-control col-6">
									  		<option>OST1</option>
									  		<option>OST2</option>
									  		<option>OST3</option>
									  		<option>OST4</option>
								      	</select>
							  		</div>
							  		
							  		<div class="form-group col-3 px-0">
								  		<select id="" class="form-control">
							  				<option>/dev/sdb</option>
							      		</select>
							  		</div>
							  		
							      	<div class="form-group col-3 px-0">
								  		<small>100GB</small>
							  		</div>
							      	
							      	<div class="col-3 text-right">
							      		<button type="button" class="btn btn-primary">activate</button>
							      		<button type="button" class="btn btn-primary">deactivate</button>
							      	</div>
							      	
							  	</div>
						  	</div>
						  </div>
						  <div class="tab-pane fade" id="profile" role="tabpanel" aria-labelledby="profile-tab">
						  
						  </div>
						  <div class="border border-top-0 col-12 px-3 py-3 text-right">
							<button id="oss-submit" type="button" class="btn btn-primary">Apply</button>
							<button id="oss-reset" type="button" class="btn btn-primary">Reset</button>
						  </div>
						</div>
					</div>
				</div>
				
			</div>
		</div>
	</div>
</body>


<jsp:include page="commons/javascropts.jsp" flush="true"></jsp:include>
<jsp:include page="commons/modals.jsp" flush="true"></jsp:include>
</html>