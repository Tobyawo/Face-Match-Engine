pipeline {
    agent any

    tools {
        // Install the Maven version configured as "M3" and add it to the path.
        maven "M3"
    }

    stages {
        stage('Build') {
            steps {
              checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: '5864ecc3-5baf-414f-9e77-a7628282a970', url: 'https://github.com/Tobyawo/gm-access-service']]])
                // Run Maven on a Unix agent.
                sh "mvn clean install"

            }
        }
           stage("Build  docker image "){
            steps{
                script{
                     sh 'docker build -t tobyawo/face-match-engine .'
                }
            }
        }

    stage("Push image to hub") {
        steps {
            script {
                // Use 'withCredentials' to access both username and password
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    // Login to Docker Hub using the credentials
                    sh '''
                    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
                    docker push tobyawo/face-match-engine
                    '''
                }
            }
        }
    }
    }
}
