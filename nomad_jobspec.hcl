job "quantsystem" {
  group "dashboard" {
    network {
      port "dashboard_port" {
        to = 8050
      }
    }
    
    task "dashboard" {
      driver = "docker"

      config {
        image = "https://ghcr.io/spyicydev/quantsystem:latest"
        ports = ["dashboard_port"]
      }

      service {
        name = "wwm"
        port = "dashboard_port"
        tags = [
          "traefik.enable=true",
        ]

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
