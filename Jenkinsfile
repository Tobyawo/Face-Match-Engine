pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'your-git-credentials-id', url: 'https://github.com/Tobyawo/Face-Match-Engine']]])
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t tobyawo/face-match-engine .'
                }
            }
        }
        stage('Push Image to Docker Hub') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
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
