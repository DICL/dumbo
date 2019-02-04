name := "hetri"

version := "0.1"

scalaVersion := "2.11.4"

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-client" % "2.7.3",
  "com.typesafe.akka" %% "akka-remote" % "2.4.11",
  "com.esotericsoftware" % "kryo" % "3.0.3",
  "it.unimi.dsi" % "fastutil" % "8.2.1",
  "com.github.oshi" % "oshi-core" % "3.9.1",
  "me.tongfei" % "progressbar" % "0.7.2",
  "junit" % "junit" % "4.10" % Test
)