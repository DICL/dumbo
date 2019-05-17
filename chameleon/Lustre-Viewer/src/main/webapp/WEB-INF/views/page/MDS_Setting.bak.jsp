<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html>

<jsp:include page="commons/head.jsp" flush="true"></jsp:include>


<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<jsp:include page="commons/header.jsp" flush="true"></jsp:include>
			  	
				<div class="col-12 mt-5">
					<h2>MDS Setting</h2>
					<div class="col-12 border">
						<div class="col-12 px-3 py-3">
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
						
						<div class="col-12 px-3 py-3">
							<label for="mds-network-fullname">
								Network Option
							</label>
							<div class="form-group col-12">
								<div class="form-inline">
						      		<input type="text" id="network_option" class="form-control col-6">
						      	</div>
							</div>
							
						</div>
						
						<div class="col-12 px-3 py-3 text-right">
							<button type="button" id="mds-apply" class="btn btn-primary">Apply</button>
							<button type="button" id="mds-reset" class="btn btn-primary">Reset</button>
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