
name := "hetri"

version := "0.1"

scalaVersion := "2.11.4"

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-client" % "2.7.3" % "provided",
  "com.typesafe.akka" %% "akka-remote" % "2.4.11",
  "com.esotericsoftware" % "kryo" % "3.0.3",
  "it.unimi.dsi" % "fastutil" % "8.2.1",
  "com.github.oshi" % "oshi-core" % "3.9.1",
  "me.tongfei" % "progressbar" % "0.7.2",
  "junit" % "junit" % "4.10" % Test
)

test in assembly := {}

//remove main-class
packageOptions in assembly ~= { os => os filterNot {_.isInstanceOf[Package.MainClass]} }

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "reference.conf" => MergeStrategy.concat
  case x => MergeStrategy.first
}


target in assembly := baseDirectory.value / "bin"
assemblyJarName in assembly := name.value + "-" + version.value + ".jar"
