job "quantsystem" {
  datacenters = ["dc1"]

  group "dashboard" {
    network {
      port "dashboard_port" {}
    }
    
    task "dashboard" {
      driver = "docker"

      config {
        image = "ghcr.io/SpyicyDev/QuantSystem:latest"
      }

      env {
        PORT = "8050"
        # Add other necessary environment variables here
      }

      service {
        name = "quantsystem-dashboard"
        port = "dashboard_port"

        check {
          name     = "dashboard-check"
          type     = "http"
          path     = "/"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }
}
